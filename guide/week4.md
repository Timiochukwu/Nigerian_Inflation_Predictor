# Week 4: Policy Shock Simulation & Java Spring Boot API

**Goal:** Simulate the effect of a +100bps MPR shock on inflation, then build the REST API layer in Java/Spring Boot.

**Prerequisite:** Week 3 complete. VAR model estimated, IRFs and FEVD computed.

---

## Day 1 — Policy Shock Simulation Design & Implementation

**Objective:** Simulate a +100 basis point MPR shock and trace its dynamic effect on inflation using the estimated VAR.

### What is a +100bps Shock?

A +100 basis point shock means the CBN raises the MPR by 1 percentage point (e.g., from 14% to 15%). We use the estimated VAR model to trace how this shock propagates through the exchange rate and money supply to ultimately affect inflation.

### Step 1: Create the simulation script

**File: `simulation/policy_shock.py`**

```python
"""
Policy Shock Simulation — +100 basis point MPR increase.

Uses the estimated VAR model to simulate the dynamic impact of a
contractionary monetary policy shock on Nigerian inflation.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

VAR_ORDERING = ["mpr", "exchange_rate", "m2", "inflation"]


def load_and_estimate_var():
    """Load data and estimate VAR."""
    filepath = os.path.join(PROCESSED_DIR, "cleaned_data.csv")
    df = pd.read_csv(filepath, index_col="date", parse_dates=True)[VAR_ORDERING]

    orders_path = os.path.join(RESULTS_DIR, "integration_orders.csv")
    use_diff = True
    if os.path.exists(orders_path):
        orders = pd.read_csv(orders_path)
        use_diff = orders["integration_order"].max() > 0

    data = df.diff().dropna() if use_diff else df.dropna()
    model = VAR(data)
    optimal_lag = model.select_order(maxlags=12).bic
    results = model.fit(optimal_lag)
    return results, data


def simulate_mpr_shock(results, shock_size_bps=100, horizon=24):
    """
    Simulate a +100bps MPR shock.

    This uses the structural IRFs (Cholesky) to trace the effect of a
    one-time shock to MPR on all variables in the system.

    Parameters
    ----------
    results : VARResults
        Estimated VAR model.
    shock_size_bps : int
        Shock size in basis points (100 = 1 percentage point).
    horizon : int
        Simulation horizon in months.

    Returns
    -------
    dict
        Simulation results for each variable.
    """
    shock_size = shock_size_bps / 100.0  # Convert bps to percentage points

    # Get the Cholesky factor of the residual covariance matrix
    sigma_u = results.sigma_u  # Residual covariance matrix
    P = np.linalg.cholesky(sigma_u)  # Lower triangular Cholesky factor

    # Standard deviation of MPR residual
    mpr_idx = VAR_ORDERING.index("mpr")
    mpr_std = np.sqrt(sigma_u[mpr_idx, mpr_idx])

    # Scale factor: how many standard deviations is our shock?
    scale = shock_size / mpr_std

    # Get orthogonalized IRFs
    irf = results.irf(periods=horizon)

    # Scale the IRFs to our specific shock size
    # IRFs are per 1 std dev shock; we scale to our desired shock
    simulation = {}

    print("=" * 70)
    print(f"POLICY SHOCK SIMULATION: +{shock_size_bps}bps MPR")
    print("=" * 70)
    print(f"\nShock size: {shock_size} percentage points")
    print(f"MPR residual std dev: {mpr_std:.4f}")
    print(f"Scale factor: {scale:.4f} standard deviations")
    print(f"Horizon: {horizon} months\n")

    for var_idx, var_name in enumerate(VAR_ORDERING):
        # Response to MPR shock (column 0 = MPR)
        response = irf.irfs[:, var_idx, mpr_idx] * scale

        simulation[var_name] = {
            "response": response.tolist(),
            "cumulative": np.cumsum(response).tolist(),
            "peak_response": float(response[np.argmax(np.abs(response))]),
            "peak_month": int(np.argmax(np.abs(response))),
        }

        print(f"  {var_name.upper()}:")
        print(f"    Peak effect: {simulation[var_name]['peak_response']:.4f} "
              f"at month {simulation[var_name]['peak_month']}")
        print(f"    Cumulative 12m: {np.cumsum(response)[min(11, horizon-1)]:.4f}")
        print(f"    Cumulative 24m: {np.cumsum(response)[-1]:.4f}")
        print()

    return simulation


def plot_simulation(simulation, shock_size_bps=100, horizon=24):
    """Plot the simulation results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {"mpr": "#1f77b4", "exchange_rate": "#2ca02c",
              "m2": "#9467bd", "inflation": "#d62728"}
    titles = {"mpr": "MPR Response", "exchange_rate": "Exchange Rate Response",
              "m2": "M2 Response", "inflation": "Inflation Response"}

    months = range(horizon + 1)

    for i, var_name in enumerate(VAR_ORDERING):
        ax = axes[i]
        response = simulation[var_name]["response"]
        cumulative = simulation[var_name]["cumulative"]

        ax.plot(months[:len(response)], response, color=colors[var_name],
                linewidth=2, label="Period effect")
        ax.plot(months[:len(cumulative)], cumulative, color=colors[var_name],
                linewidth=2, linestyle="--", label="Cumulative", alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_title(titles[var_name], fontsize=12, fontweight="bold")
        ax.set_xlabel("Months after shock")
        ax.set_ylabel("Change")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Effect of +{shock_size_bps}bps MPR Shock on Nigerian Economy",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "simulation_mpr_shock.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: simulation_mpr_shock.png")

    # Focused plot: Inflation response only
    fig, ax = plt.subplots(figsize=(12, 5))
    inf_response = simulation["inflation"]["response"]
    inf_cumulative = simulation["inflation"]["cumulative"]

    ax.plot(months[:len(inf_response)], inf_response, color="#d62728",
            linewidth=2.5, label="Period effect on inflation")
    ax.plot(months[:len(inf_cumulative)], inf_cumulative, color="#d62728",
            linewidth=2.5, linestyle="--", label="Cumulative effect", alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.fill_between(months[:len(inf_response)], 0, inf_response,
                    alpha=0.1, color="#d62728")
    ax.set_title(f"Inflation Response to +{shock_size_bps}bps MPR Shock",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Months after shock", fontsize=12)
    ax.set_ylabel("Change in inflation (percentage points)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "simulation_inflation_response.png"),
                dpi=150)
    plt.close(fig)
    print("Saved: simulation_inflation_response.png")


def interpret_simulation(simulation, shock_size_bps=100):
    """Generate policy interpretation of the simulation."""
    print("\n" + "=" * 70)
    print("POLICY INTERPRETATION")
    print("=" * 70)

    inf = simulation["inflation"]
    exr = simulation["exchange_rate"]

    print(f"\n  Scenario: The CBN raises the MPR by {shock_size_bps} basis points.")
    print()

    # Inflation impact
    if inf["peak_response"] < 0:
        print(f"  INFLATION: Decreases by up to {abs(inf['peak_response']):.2f} percentage")
        print(f"  points at month {inf['peak_month']}.")
        print(f"  The cumulative 12-month reduction is {abs(inf['cumulative'][11]):.2f} pp.")
        print(f"  Monetary tightening is effective in reducing Nigerian inflation,")
        print(f"  though the effect is gradual and peaks after {inf['peak_month']} months.")
    else:
        print(f"  INFLATION: Initially INCREASES by {inf['peak_response']:.2f} pp")
        print(f"  (price puzzle). This may reflect:")
        print(f"  - Cost-push effects from higher borrowing costs")
        print(f"  - Supply-side constraints in the Nigerian economy")
        print(f"  - Weak monetary transmission channels")

    # Exchange rate impact
    print()
    if exr["peak_response"] < 0:
        print(f"  EXCHANGE RATE: Naira appreciates (by {abs(exr['peak_response']):.2f} units)")
        print(f"  following the rate hike — consistent with interest rate parity.")
    else:
        print(f"  EXCHANGE RATE: Naira depreciates despite rate hike —")
        print(f"  may indicate capital flight concerns or credibility issues.")

    print(f"\n  POLICY IMPLICATION:")
    print(f"  The CBN should expect the inflation effect of a {shock_size_bps}bps")
    print(f"  rate hike to fully materialize over {inf['peak_month']}-24 months.")
    print(f"  This lag must be factored into MPC decision-making.")


def save_simulation(simulation, shock_size_bps):
    """Save simulation results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output = {
        "shock_size_bps": shock_size_bps,
        "variables": {}
    }
    for var, data in simulation.items():
        output["variables"][var] = {
            "peak_response": data["peak_response"],
            "peak_month": data["peak_month"],
            "cumulative_12m": data["cumulative"][min(11, len(data["cumulative"])-1)],
            "cumulative_24m": data["cumulative"][-1],
        }

    with open(os.path.join(RESULTS_DIR, "simulation_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: simulation_results.json")


if __name__ == "__main__":
    results, data = load_and_estimate_var()
    simulation = simulate_mpr_shock(results, shock_size_bps=100, horizon=24)
    plot_simulation(simulation, shock_size_bps=100, horizon=24)
    interpret_simulation(simulation, shock_size_bps=100)
    save_simulation(simulation, shock_size_bps=100)
    print("\nSimulation complete.")
```

### Step 2: Run

```bash
python -m simulation.policy_shock
```

### Step 3: Commit

```bash
git add simulation/policy_shock.py
git commit -m "Week 4 Day 1: Add +100bps MPR shock simulation with plots and interpretation"
```

---

## Day 2 — Java Spring Boot API Setup

**Objective:** Initialize the Spring Boot project for the REST API layer.

### Step 1: Install Java 17 and Maven (if not installed)

```bash
# Check Java version
java -version

# On Ubuntu/Debian:
sudo apt update && sudo apt install openjdk-17-jdk maven -y

# On Mac:
brew install openjdk@17 maven
```

### Step 2: Generate the Spring Boot project

Use Spring Initializr or create manually:

```bash
cd api/
```

Create the Maven project structure:

```bash
mkdir -p src/main/java/com/nigerianinflation/api/controller
mkdir -p src/main/java/com/nigerianinflation/api/service
mkdir -p src/main/java/com/nigerianinflation/api/model
mkdir -p src/main/java/com/nigerianinflation/api/repository
mkdir -p src/main/resources
mkdir -p src/test/java/com/nigerianinflation/api
```

### Step 3: Create `pom.xml`

**File: `api/pom.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.1</version>
    </parent>

    <groupId>com.nigerianinflation</groupId>
    <artifactId>inflation-predictor-api</artifactId>
    <version>1.0.0</version>
    <name>Nigerian Inflation Predictor API</name>
    <description>REST API for Nigerian Inflation Predictor model results</description>

    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <!-- Spring Boot Web -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Spring Boot JPA (PostgreSQL) -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>

        <!-- PostgreSQL Driver -->
        <dependency>
            <groupId>org.postgresql</groupId>
            <artifactId>postgresql</artifactId>
            <scope>runtime</scope>
        </dependency>

        <!-- Jackson JSON -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
        </dependency>

        <!-- Test -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
```

### Step 4: Create the main application class

**File: `api/src/main/java/com/nigerianinflation/api/Application.java`**

```java
package com.nigerianinflation.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

### Step 5: Create `application.properties`

**File: `api/src/main/resources/application.properties`**

```properties
server.port=8080

# PostgreSQL
spring.datasource.url=jdbc:postgresql://localhost:5432/nigerian_inflation
spring.datasource.username=postgres
spring.datasource.password=postgres
spring.datasource.driver-class-name=org.postgresql.Driver

# JPA
spring.jpa.hibernate.ddl-auto=validate
spring.jpa.show-sql=false
spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.PostgreSQLDialect

# JSON
spring.jackson.serialization.write-dates-as-timestamps=false
```

### Step 6: Build and verify

```bash
cd api/
mvn clean compile
```

If it compiles without errors, the project is set up correctly.

### Step 7: Commit

```bash
cd ..
git add api/
git commit -m "Week 4 Day 2: Initialize Spring Boot API project"
```

---

## Day 3 — API Entity Models

**Objective:** Create JPA entity classes that map to the PostgreSQL tables.

### Step 1: Create the MacroMonthly entity

**File: `api/src/main/java/com/nigerianinflation/api/model/MacroMonthly.java`**

```java
package com.nigerianinflation.api.model;

import jakarta.persistence.*;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "macro_monthly")
public class MacroMonthly {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private LocalDate date;

    @Column(nullable = false)
    private Double mpr;

    @Column(nullable = false)
    private Double inflation;

    @Column(name = "exchange_rate", nullable = false)
    private Double exchangeRate;

    @Column(nullable = false)
    private Double m2;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }

    public Double getMpr() { return mpr; }
    public void setMpr(Double mpr) { this.mpr = mpr; }

    public Double getInflation() { return inflation; }
    public void setInflation(Double inflation) { this.inflation = inflation; }

    public Double getExchangeRate() { return exchangeRate; }
    public void setExchangeRate(Double exchangeRate) { this.exchangeRate = exchangeRate; }

    public Double getM2() { return m2; }
    public void setM2(Double m2) { this.m2 = m2; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
```

### Step 2: Create the ModelResult entity

**File: `api/src/main/java/com/nigerianinflation/api/model/ModelResult.java`**

```java
package com.nigerianinflation.api.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "model_results")
public class ModelResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "model_name", nullable = false)
    private String modelName;

    @Column(name = "result_type", nullable = false)
    private String resultType;

    @Column(columnDefinition = "jsonb")
    private String parameters;

    @Column(name = "result_data", nullable = false, columnDefinition = "jsonb")
    private String resultData;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getModelName() { return modelName; }
    public void setModelName(String modelName) { this.modelName = modelName; }

    public String getResultType() { return resultType; }
    public void setResultType(String resultType) { this.resultType = resultType; }

    public String getParameters() { return parameters; }
    public void setParameters(String parameters) { this.parameters = parameters; }

    public String getResultData() { return resultData; }
    public void setResultData(String resultData) { this.resultData = resultData; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
}
```

### Step 3: Create repositories

**File: `api/src/main/java/com/nigerianinflation/api/repository/MacroMonthlyRepository.java`**

```java
package com.nigerianinflation.api.repository;

import com.nigerianinflation.api.model.MacroMonthly;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;

@Repository
public interface MacroMonthlyRepository extends JpaRepository<MacroMonthly, Long> {
    List<MacroMonthly> findByDateBetweenOrderByDate(LocalDate start, LocalDate end);
    MacroMonthly findTopByOrderByDateDesc();
}
```

**File: `api/src/main/java/com/nigerianinflation/api/repository/ModelResultRepository.java`**

```java
package com.nigerianinflation.api.repository;

import com.nigerianinflation.api.model.ModelResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface ModelResultRepository extends JpaRepository<ModelResult, Long> {
    List<ModelResult> findByModelName(String modelName);
    List<ModelResult> findByResultType(String resultType);
    ModelResult findTopByModelNameOrderByCreatedAtDesc(String modelName);
}
```

### Step 4: Commit

```bash
git add api/
git commit -m "Week 4 Day 3: Add JPA entity models and repositories"
```

---

## Day 4 — API Service Layer & Controllers

**Objective:** Create service classes and REST controllers to expose model results.

### Step 1: Create the service

**File: `api/src/main/java/com/nigerianinflation/api/service/InflationService.java`**

```java
package com.nigerianinflation.api.service;

import com.nigerianinflation.api.model.MacroMonthly;
import com.nigerianinflation.api.model.ModelResult;
import com.nigerianinflation.api.repository.MacroMonthlyRepository;
import com.nigerianinflation.api.repository.ModelResultRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.util.List;

@Service
public class InflationService {

    private final MacroMonthlyRepository macroRepo;
    private final ModelResultRepository modelRepo;

    public InflationService(MacroMonthlyRepository macroRepo,
                            ModelResultRepository modelRepo) {
        this.macroRepo = macroRepo;
        this.modelRepo = modelRepo;
    }

    public List<MacroMonthly> getAllData() {
        return macroRepo.findAll();
    }

    public List<MacroMonthly> getDataByRange(LocalDate start, LocalDate end) {
        return macroRepo.findByDateBetweenOrderByDate(start, end);
    }

    public MacroMonthly getLatestData() {
        return macroRepo.findTopByOrderByDateDesc();
    }

    public List<ModelResult> getModelResults(String modelName) {
        return modelRepo.findByModelName(modelName);
    }

    public ModelResult getLatestResult(String modelName) {
        return modelRepo.findTopByModelNameOrderByCreatedAtDesc(modelName);
    }

    public List<ModelResult> getResultsByType(String resultType) {
        return modelRepo.findByResultType(resultType);
    }
}
```

### Step 2: Create the REST controllers

**File: `api/src/main/java/com/nigerianinflation/api/controller/DataController.java`**

```java
package com.nigerianinflation.api.controller;

import com.nigerianinflation.api.model.MacroMonthly;
import com.nigerianinflation.api.service.InflationService;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.List;

@RestController
@RequestMapping("/api/v1/data")
public class DataController {

    private final InflationService service;

    public DataController(InflationService service) {
        this.service = service;
    }

    @GetMapping
    public ResponseEntity<List<MacroMonthly>> getAllData() {
        return ResponseEntity.ok(service.getAllData());
    }

    @GetMapping("/range")
    public ResponseEntity<List<MacroMonthly>> getDataByRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate start,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate end) {
        return ResponseEntity.ok(service.getDataByRange(start, end));
    }

    @GetMapping("/latest")
    public ResponseEntity<MacroMonthly> getLatest() {
        return ResponseEntity.ok(service.getLatestData());
    }
}
```

**File: `api/src/main/java/com/nigerianinflation/api/controller/ModelController.java`**

```java
package com.nigerianinflation.api.controller;

import com.nigerianinflation.api.model.ModelResult;
import com.nigerianinflation.api.service.InflationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/v1/models")
public class ModelController {

    private final InflationService service;

    public ModelController(InflationService service) {
        this.service = service;
    }

    @GetMapping("/{modelName}")
    public ResponseEntity<List<ModelResult>> getModelResults(
            @PathVariable String modelName) {
        return ResponseEntity.ok(service.getModelResults(modelName));
    }

    @GetMapping("/{modelName}/latest")
    public ResponseEntity<ModelResult> getLatestResult(
            @PathVariable String modelName) {
        return ResponseEntity.ok(service.getLatestResult(modelName));
    }

    @GetMapping("/type/{resultType}")
    public ResponseEntity<List<ModelResult>> getByType(
            @PathVariable String resultType) {
        return ResponseEntity.ok(service.getResultsByType(resultType));
    }
}
```

### Step 3: Build

```bash
cd api/
mvn clean compile
cd ..
```

### Step 4: Commit

```bash
git add api/
git commit -m "Week 4 Day 4: Add API service layer and REST controllers"
```

---

## Day 5 — Store Python Results in PostgreSQL for API

**Objective:** Create a Python script that stores all model results (ARDL, VAR, IRF, FEVD, simulation) into the PostgreSQL `model_results` table so the API can serve them.

### Step 1: Create the results storage script

**File: `data_ingestion/store_results.py`**

```python
"""
Store all model results in PostgreSQL for API consumption.

Reads JSON/CSV results from the results/ folder and stores them
in the model_results table.
"""

import os
import json
import pandas as pd
from sqlalchemy import create_engine, text

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": os.environ.get("DB_PORT", "5432"),
    "database": os.environ.get("DB_NAME", "nigerian_inflation"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "postgres"),
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def get_engine():
    url = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(url)


def store_result(engine, model_name, result_type, result_data, parameters=None):
    """Insert a model result into the database."""
    with engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO model_results (model_name, result_type, parameters, result_data)
                VALUES (:model_name, :result_type, :parameters, :result_data)
            """),
            {
                "model_name": model_name,
                "result_type": result_type,
                "parameters": json.dumps(parameters) if parameters else None,
                "result_data": json.dumps(result_data),
            }
        )
        conn.commit()
    print(f"  Stored: {model_name} / {result_type}")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def store_all_results():
    """Store all computed results in the database."""
    engine = get_engine()

    # Clear existing results
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM model_results"))
        conn.commit()

    print("Storing model results in PostgreSQL...")

    # 1. Stationarity tests
    data = load_json("stationarity_tests.json")
    if data:
        store_result(engine, "stationarity", "adf_kpss", data)

    # 2. Integration orders
    path = os.path.join(RESULTS_DIR, "integration_orders.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        store_result(engine, "stationarity", "integration_orders",
                     df.to_dict(orient="records"))

    # 3. ARDL results
    data = load_json("ardl_metrics.json")
    if data:
        store_result(engine, "ardl", "metrics", data)

    data = load_json("ardl_ecm_results.json")
    if data:
        store_result(engine, "ardl", "ecm", data)

    # 4. Bounds test
    data = load_json("bounds_test.json")
    if data:
        store_result(engine, "ardl", "bounds_test", data)

    # 5. ARDL diagnostics
    data = load_json("ardl_diagnostics.json")
    if data:
        store_result(engine, "ardl", "diagnostics", data)

    # 6. VAR metrics
    data = load_json("var_metrics.json")
    if data:
        store_result(engine, "var", "metrics", data)

    # 7. VAR diagnostics
    data = load_json("var_diagnostics.json")
    if data:
        store_result(engine, "var", "diagnostics", data)

    # 8. IRF interpretation
    data = load_json("irf_interpretation.json")
    if data:
        store_result(engine, "var", "irf", data)

    # 9. FEVD
    path = os.path.join(RESULTS_DIR, "fevd_inflation.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        store_result(engine, "var", "fevd", df.to_dict(orient="records"))

    # 10. Simulation
    data = load_json("simulation_results.json")
    if data:
        store_result(engine, "simulation", "mpr_shock_100bps", data)

    # Verify
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM model_results")).scalar()
    print(f"\nTotal results stored: {count}")


if __name__ == "__main__":
    store_all_results()
```

### Step 2: Run

```bash
python -m data_ingestion.store_results
```

### Step 3: Test the API

```bash
cd api/
mvn spring-boot:run
```

In another terminal:
```bash
# Get all macro data
curl http://localhost:8080/api/v1/data/latest

# Get ARDL results
curl http://localhost:8080/api/v1/models/ardl

# Get simulation results
curl http://localhost:8080/api/v1/models/simulation/latest
```

### Step 4: Commit

```bash
cd ..
git add data_ingestion/store_results.py
git commit -m "Week 4 Day 5: Add Python-to-PostgreSQL results storage for API"
```

### What You Know After Week 4

- How a +100bps MPR shock affects inflation over 24 months
- The peak effect timing and magnitude
- How to build a Spring Boot API that serves econometric results
- The full data flow: Python models -> PostgreSQL -> Java API -> JSON

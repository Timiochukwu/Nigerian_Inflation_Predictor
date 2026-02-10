# Day 33 — PostgreSQL: Storing Results in a Database

## What You'll Learn Today

So far, our econometric results live in JSON files on disk. That's fine for learning, but real applications need a **database** because:

1. **Persistence**: Data survives server restarts
2. **Querying**: SQL lets you filter, sort, and aggregate results efficiently
3. **Concurrent Access**: Multiple users can read/write without file-locking issues
4. **Relationships**: Connect forecasts to their underlying VAR parameters, diagnostics, etc.

Today you'll install **PostgreSQL** (a popular open-source relational database), create JPA entities (Java classes that map to database tables), and write a data loader that reads your Python JSON results and stores them in the database.

---

## Step 1: Install PostgreSQL

### On Ubuntu/Debian

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### On macOS (with Homebrew)

```bash
brew install postgresql
brew services start postgresql
```

### On Windows

Download the installer from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/) and follow the wizard. Remember the password you set for the `postgres` user.

---

## Step 2: Create the Database

Switch to the `postgres` user and create a database called `nip_db`:

```bash
sudo -u postgres psql
```

Inside the PostgreSQL shell:

```sql
CREATE DATABASE nip_db;
CREATE USER nip_user WITH ENCRYPTED PASSWORD 'nip_pass';
GRANT ALL PRIVILEGES ON DATABASE nip_db TO nip_user;
\q
```

(On Windows, open SQL Shell (psql) and log in as `postgres`, then run the same SQL commands.)

**What just happened?**

- Created a database `nip_db`
- Created a user `nip_user` with password `nip_pass`
- Granted full permissions to that user

---

## Step 3: Add PostgreSQL Dependencies

Open `backend/pom.xml` and add the PostgreSQL driver and Spring Data JPA (if not already present):

**Delete everything in `backend/pom.xml` and replace it with this:**

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
        <version>3.2.2</version>
        <relativePath/>
    </parent>

    <groupId>com.nip</groupId>
    <artifactId>backend</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>Nigerian Inflation Predictor Backend</name>
    <description>Spring Boot backend for Nigerian Inflation Predictor</description>

    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <!-- Spring Boot Web -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Spring Data JPA -->
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

        <!-- Spring Boot DevTools -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>

        <!-- Spring Boot Test -->
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

---

## Step 4: Configure Database Connection

Open `backend/src/main/resources/application.properties` and add database settings:

**Delete everything in `backend/src/main/resources/application.properties` and replace it with this:**

```properties
# Server port
server.port=8080

# Database connection
spring.datasource.url=jdbc:postgresql://localhost:5432/nip_db
spring.datasource.username=nip_user
spring.datasource.password=nip_pass
spring.datasource.driver-class-name=org.postgresql.Driver

# JPA/Hibernate
spring.jpa.database-platform=org.hibernate.dialect.PostgreSQLDialect
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true

# Jackson JSON
spring.jackson.serialization.write-dates-as-timestamps=false
```

**What does `ddl-auto=update` mean?**

Hibernate will automatically create/update database tables based on your entity classes. Perfect for development. In production, you'd use migrations (Flyway/Liquibase).

---

## Step 5: Create JPA Entity Classes

We'll create two entities:

1. **EconometricResult**: Stores metadata (date, model type, lag order)
2. **ForecastEntry**: Stores individual forecast points (date, value)

### 5a. Create `EconometricResult.java`

**Create file `backend/src/main/java/com/nip/backend/model/EconometricResult.java` with this content:**

```java
package com.nip.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "econometric_results")
public class EconometricResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String modelType;  // "VAR", "VECM", etc.

    @Column(nullable = false)
    private Integer lagOrder;

    @Column(nullable = false)
    private LocalDateTime createdAt;

    @Column(columnDefinition = "TEXT")
    private String diagnostics;  // JSON string with AIC, BIC, etc.

    @OneToMany(mappedBy = "result", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<ForecastEntry> forecasts = new ArrayList<>();

    // Constructors
    public EconometricResult() {
        this.createdAt = LocalDateTime.now();
    }

    public EconometricResult(String modelType, Integer lagOrder) {
        this.modelType = modelType;
        this.lagOrder = lagOrder;
        this.createdAt = LocalDateTime.now();
    }

    // Getters and setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public Integer getLagOrder() {
        return lagOrder;
    }

    public void setLagOrder(Integer lagOrder) {
        this.lagOrder = lagOrder;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }

    public String getDiagnostics() {
        return diagnostics;
    }

    public void setDiagnostics(String diagnostics) {
        this.diagnostics = diagnostics;
    }

    public List<ForecastEntry> getForecasts() {
        return forecasts;
    }

    public void setForecasts(List<ForecastEntry> forecasts) {
        this.forecasts = forecasts;
    }

    // Helper method to add forecast
    public void addForecast(ForecastEntry forecast) {
        forecasts.add(forecast);
        forecast.setResult(this);
    }
}
```

**Key annotations:**

- `@Entity`: Tells JPA this is a database table
- `@Table(name = "...")`: Custom table name
- `@Id` + `@GeneratedValue`: Auto-incrementing primary key
- `@OneToMany`: One result has many forecast entries
- `cascade = CascadeType.ALL`: When you save the result, forecasts save too

### 5b. Create `ForecastEntry.java`

**Create file `backend/src/main/java/com/nip/backend/model/ForecastEntry.java` with this content:**

```java
package com.nip.backend.model;

import jakarta.persistence.*;
import java.time.LocalDate;

@Entity
@Table(name = "forecast_entries")
public class ForecastEntry {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private LocalDate forecastDate;

    @Column(nullable = false)
    private Double forecastValue;

    @Column
    private Double lowerBound;

    @Column
    private Double upperBound;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "result_id", nullable = false)
    private EconometricResult result;

    // Constructors
    public ForecastEntry() {}

    public ForecastEntry(LocalDate forecastDate, Double forecastValue) {
        this.forecastDate = forecastDate;
        this.forecastValue = forecastValue;
    }

    // Getters and setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public LocalDate getForecastDate() {
        return forecastDate;
    }

    public void setForecastDate(LocalDate forecastDate) {
        this.forecastDate = forecastDate;
    }

    public Double getForecastValue() {
        return forecastValue;
    }

    public void setForecastValue(Double forecastValue) {
        this.forecastValue = forecastValue;
    }

    public Double getLowerBound() {
        return lowerBound;
    }

    public void setLowerBound(Double lowerBound) {
        this.lowerBound = lowerBound;
    }

    public Double getUpperBound() {
        return upperBound;
    }

    public void setUpperBound(Double upperBound) {
        this.upperBound = upperBound;
    }

    public EconometricResult getResult() {
        return result;
    }

    public void setResult(EconometricResult result) {
        this.result = result;
    }
}
```

**Key annotations:**

- `@ManyToOne`: Many forecast entries belong to one result
- `@JoinColumn`: Defines the foreign key column

---

## Step 6: Create Repositories

Spring Data JPA lets you write repositories with **zero SQL**. Just define an interface extending `JpaRepository` and you get `save()`, `findAll()`, `findById()`, etc. for free.

### 6a. Create `EconometricResultRepository.java`

**Create file `backend/src/main/java/com/nip/backend/repository/EconometricResultRepository.java` with this content:**

```java
package com.nip.backend.repository;

import com.nip.backend.model.EconometricResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface EconometricResultRepository extends JpaRepository<EconometricResult, Long> {

    // Custom query methods (Spring generates SQL automatically)
    List<EconometricResult> findByModelType(String modelType);

    List<EconometricResult> findByModelTypeOrderByCreatedAtDesc(String modelType);
}
```

**What's happening?**

- `JpaRepository<EconometricResult, Long>`: Manages `EconometricResult` entities with `Long` primary keys
- Spring sees `findByModelType` and generates: `SELECT * FROM econometric_results WHERE model_type = ?`
- `OrderByCreatedAtDesc`: Adds `ORDER BY created_at DESC`

### 6b. Create `ForecastEntryRepository.java`

**Create file `backend/src/main/java/com/nip/backend/repository/ForecastEntryRepository.java` with this content:**

```java
package com.nip.backend.repository;

import com.nip.backend.model.ForecastEntry;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ForecastEntryRepository extends JpaRepository<ForecastEntry, Long> {
    // Inherits basic CRUD methods
}
```

---

## Step 7: Create a Data Loader

Now let's load your Python JSON results into the database. We'll create a `DataLoader` component that runs on startup and imports `output/var_forecast.json`.

**Create file `backend/src/main/java/com/nip/backend/loader/DataLoader.java` with this content:**

```java
package com.nip.backend.loader;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nip.backend.model.EconometricResult;
import com.nip.backend.model.ForecastEntry;
import com.nip.backend.repository.EconometricResultRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import java.io.File;
import java.time.LocalDate;

@Component
public class DataLoader implements CommandLineRunner {

    private final EconometricResultRepository resultRepository;
    private final ObjectMapper objectMapper;

    public DataLoader(EconometricResultRepository resultRepository) {
        this.resultRepository = resultRepository;
        this.objectMapper = new ObjectMapper();
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("=== DataLoader: Loading JSON results into database ===");

        // Path to your Python output
        File jsonFile = new File("output/var_forecast.json");
        if (!jsonFile.exists()) {
            System.out.println("⚠️  output/var_forecast.json not found. Skipping data load.");
            return;
        }

        // Parse JSON
        JsonNode root = objectMapper.readTree(jsonFile);

        // Create EconometricResult entity
        EconometricResult result = new EconometricResult("VAR", 2);

        // Add diagnostics (optional)
        if (root.has("aic")) {
            String diagnostics = String.format(
                "{\"aic\": %s, \"bic\": %s}",
                root.get("aic").asText(),
                root.get("bic").asText()
            );
            result.setDiagnostics(diagnostics);
        }

        // Parse forecast array
        if (root.has("forecast")) {
            JsonNode forecastArray = root.get("forecast");
            for (JsonNode item : forecastArray) {
                String dateStr = item.get("date").asText();
                double value = item.get("infl").asDouble();

                LocalDate forecastDate = LocalDate.parse(dateStr);
                ForecastEntry entry = new ForecastEntry(forecastDate, value);
                result.addForecast(entry);
            }
        }

        // Save to database (cascades to forecast entries)
        resultRepository.save(result);

        System.out.println("✅ Loaded " + result.getForecasts().size() + " forecast entries into database");
    }
}
```

**What's happening?**

- `CommandLineRunner`: Spring calls `run()` on startup
- Jackson reads `var_forecast.json`
- Creates `EconometricResult` and links `ForecastEntry` objects
- `resultRepository.save(result)` saves everything (thanks to cascade)

---

## Step 8: Run the Application

From the `backend` directory:

```bash
mvn clean install
mvn spring-boot:run
```

**Expected output:**

```
=== DataLoader: Loading JSON results into database ===
Hibernate: create table econometric_results (...)
Hibernate: create table forecast_entries (...)
Hibernate: insert into econometric_results (...)
Hibernate: insert into forecast_entries (...)
✅ Loaded 12 forecast entries into database
Started BackendApplication in 3.456 seconds
```

---

## Step 9: Verify Data in PostgreSQL

Open a PostgreSQL shell:

```bash
psql -U nip_user -d nip_db
```

Run queries:

```sql
-- List results
SELECT id, model_type, lag_order, created_at FROM econometric_results;

-- List forecasts
SELECT id, forecast_date, forecast_value, result_id FROM forecast_entries;

-- Join to see forecasts with their parent result
SELECT r.model_type, r.lag_order, f.forecast_date, f.forecast_value
FROM econometric_results r
JOIN forecast_entries f ON r.id = f.result_id
ORDER BY f.forecast_date;
```

You should see your VAR forecasts stored as rows!

---

## Step 10: Commit Your Work

```bash
git add backend/pom.xml
git add backend/src/main/resources/application.properties
git add backend/src/main/java/com/nip/backend/model/
git add backend/src/main/java/com/nip/backend/repository/
git add backend/src/main/java/com/nip/backend/loader/
git commit -m "Day 33: Add PostgreSQL persistence with JPA entities, repositories, and data loader

- Configure PostgreSQL connection in application.properties
- Create EconometricResult and ForecastEntry entities
- Implement JPA repositories for database access
- Add DataLoader to import JSON results on startup
- Verify forecasts stored in database

https://claude.ai/code/session_0173mGraLvrZwx6v1oTi7RnN"
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## Common Errors and Fixes

### Error: "org.postgresql.util.PSQLException: FATAL: password authentication failed"

**Fix**: Check your `application.properties`. Make sure `nip_user` and `nip_pass` match what you created in Step 2.

### Error: "relation 'econometric_results' does not exist"

**Fix**: Hibernate should auto-create tables with `ddl-auto=update`. If it doesn't, check:
1. `spring.jpa.hibernate.ddl-auto=update` is set
2. The entity classes have `@Entity` annotations
3. Restart the application

### Error: "com.fasterxml.jackson.databind.exc.InvalidFormatException"

**Fix**: Your JSON date format doesn't match Java's `LocalDate.parse()`. Ensure dates are "YYYY-MM-DD" in the JSON, or add a custom deserializer.

### Application starts but no data loads

**Fix**: Check the file path in `DataLoader.java`. If you're running from `backend/`, the path should be `../output/var_forecast.json` (relative to the backend directory). Update:

```java
File jsonFile = new File("../output/var_forecast.json");
```

---

## Q&A

**Q: Why PostgreSQL instead of MySQL or MongoDB?**

A: PostgreSQL has excellent support for JSON, time-series data, and complex queries. It's free, open-source, and widely used in production. MySQL is also fine. MongoDB (NoSQL) doesn't fit well with JPA.

**Q: What's the difference between `@OneToMany` and `@ManyToOne`?**

A: They're two sides of the same relationship. `EconometricResult` has `@OneToMany` forecasts (one result has many entries). `ForecastEntry` has `@ManyToOne` result (many entries belong to one result).

**Q: Should I commit the database password to Git?**

A: **NO!** For production, use environment variables:

```properties
spring.datasource.password=${DB_PASSWORD}
```

Then set `DB_PASSWORD` in your environment. For learning, hardcoding is okay (just don't use the same password elsewhere).

**Q: Can I query the database from Python instead of Java?**

A: Yes! Use `psycopg2` or `sqlalchemy`. But the point of Spring Boot is to build a backend that serves data via REST APIs (coming in Day 34).

**Q: What if I want to store VAR coefficients, not just forecasts?**

A: Create another entity, e.g., `VarCoefficient`, with fields like `variable`, `lag`, `coefficient`. Add `@OneToMany` relationship to `EconometricResult`.

**Q: Do I need to restart the app every time I change an entity?**

A: With `spring-boot-devtools`, it auto-restarts on file changes. But if you change the database schema (add/remove columns), you may need to manually alter tables or drop/recreate them.

---

## What's Next?

- **Day 34**: Build REST endpoints to query forecasts from the database
- **Day 35**: Add authentication and authorization (only admins can write)
- **Day 36**: Deploy to a cloud database (AWS RDS or Heroku Postgres)

You now have a **persistent backend** that stores econometric results in a professional relational database. Tomorrow you'll expose this data via REST APIs so your frontend can fetch it!

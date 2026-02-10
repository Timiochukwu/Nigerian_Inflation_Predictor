# Day 32 — REST Endpoints: Serving Econometric Results

## What You'll Learn Today

Today you'll learn how to create REST API endpoints that serve your econometric results to the frontend. A REST API is how your backend (Spring Boot) communicates with your frontend (React). When a user visits your dashboard, the frontend will make HTTP requests to these endpoints to fetch data.

**What you'll build:**
- Model classes (POJOs) to represent your data
- A service that reads JSON files from Python's `results/` directory
- A controller with three endpoints:
  - `GET /api/results/summary` — returns VAR summary statistics
  - `GET /api/results/irf` — returns impulse response function data
  - `GET /api/results/forecast` — returns forecast predictions

By the end of today, you'll be able to type `curl http://localhost:8080/api/results/summary` and see your econometric results as JSON.

---

## Key Concepts

### What is REST?

REST (Representational State Transfer) is a way for applications to communicate over HTTP. Think of it like a menu at a restaurant:
- The menu lists what you can order (endpoints)
- You make a request (GET, POST, etc.)
- The kitchen prepares it (your service layer)
- You get your food (JSON response)

### HTTP Methods

- **GET** — retrieve data (read-only, no side effects)
- **POST** — send data to create something new
- **PUT** — update existing data
- **DELETE** — remove data

Today we'll only use GET because we're just reading files.

### JSON Response

Your endpoints will return JSON (JavaScript Object Notation), which looks like this:

```json
{
  "aic": -14.23,
  "bic": -13.87,
  "observations": 120
}
```

Spring Boot automatically converts Java objects to JSON when you return them from a `@RestController`.

---

## Step 1: Create the Model Package

First, let's organize our code. Create a new package for your data models:

```bash
mkdir -p api/src/main/java/com/nip/model
```

Now you'll create three model classes to represent your econometric results.

---

## Step 2: Create VARSummary Model

**Delete everything in `api/src/main/java/com/nip/model/VARSummary.java` and replace it with this:**

```java
package com.nip.model;

public class VARSummary {
    private double aic;
    private double bic;
    private double fpe;
    private double hqic;
    private int observations;
    private int lags;
    private String[] variables;

    // Default constructor (required for JSON deserialization)
    public VARSummary() {
    }

    // Constructor with all fields
    public VARSummary(double aic, double bic, double fpe, double hqic,
                      int observations, int lags, String[] variables) {
        this.aic = aic;
        this.bic = bic;
        this.fpe = fpe;
        this.hqic = hqic;
        this.observations = observations;
        this.lags = lags;
        this.variables = variables;
    }

    // Getters and setters
    public double getAic() {
        return aic;
    }

    public void setAic(double aic) {
        this.aic = aic;
    }

    public double getBic() {
        return bic;
    }

    public void setBic(double bic) {
        this.bic = bic;
    }

    public double getFpe() {
        return fpe;
    }

    public void setFpe(double fpe) {
        this.fpe = fpe;
    }

    public double getHqic() {
        return hqic;
    }

    public void setHqic(double hqic) {
        this.hqic = hqic;
    }

    public int getObservations() {
        return observations;
    }

    public void setObservations(int observations) {
        this.observations = observations;
    }

    public int getLags() {
        return lags;
    }

    public void setLags(int lags) {
        this.lags = lags;
    }

    public String[] getVariables() {
        return variables;
    }

    public void setVariables(String[] variables) {
        this.variables = variables;
    }
}
```

**What this does:**
- A POJO (Plain Old Java Object) that holds VAR model summary statistics
- `aic`, `bic`, `fpe`, `hqic` are information criteria (lower is better)
- `observations` is the number of data points used
- `lags` is the optimal lag order
- `variables` is the array of variable names (MPR, TBR, EXO, INF)
- Spring Boot will automatically convert this to JSON when you return it

---

## Step 3: Create IRFData Model

**Delete everything in `api/src/main/java/com/nip/model/IRFData.java` and replace it with this:**

```java
package com.nip.model;

public class IRFData {
    private String shock;      // e.g., "MPR"
    private String response;   // e.g., "INF"
    private int[] periods;     // [0, 1, 2, 3, ...]
    private double[] values;   // [0.0, 0.12, 0.34, ...]
    private double[] lower;    // Lower confidence band
    private double[] upper;    // Upper confidence band

    // Default constructor
    public IRFData() {
    }

    // Constructor with all fields
    public IRFData(String shock, String response, int[] periods,
                   double[] values, double[] lower, double[] upper) {
        this.shock = shock;
        this.response = response;
        this.periods = periods;
        this.values = values;
        this.lower = lower;
        this.upper = upper;
    }

    // Getters and setters
    public String getShock() {
        return shock;
    }

    public void setShock(String shock) {
        this.shock = shock;
    }

    public String getResponse() {
        return response;
    }

    public void setResponse(String response) {
        this.response = response;
    }

    public int[] getPeriods() {
        return periods;
    }

    public void setPeriods(int[] periods) {
        this.periods = periods;
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        this.values = values;
    }

    public double[] getLower() {
        return lower;
    }

    public void setLower(double[] lower) {
        this.lower = lower;
    }

    public double[] getUpper() {
        return upper;
    }

    public void setUpper(double[] upper) {
        this.upper = upper;
    }
}
```

**What this does:**
- Represents one impulse-response function (e.g., MPR shock → INF response)
- `periods` is the x-axis (0, 1, 2, ... months ahead)
- `values` is the y-axis (the response values)
- `lower` and `upper` are confidence bands (for error bars on charts)

---

## Step 4: Create ForecastData Model

**Delete everything in `api/src/main/java/com/nip/model/ForecastData.java` and replace it with this:**

```java
package com.nip.model;

public class ForecastData {
    private String variable;     // e.g., "INF"
    private String[] dates;      // ["2025-01", "2025-02", ...]
    private double[] forecast;   // Predicted values
    private double[] lower;      // Lower confidence bound
    private double[] upper;      // Upper confidence bound

    // Default constructor
    public ForecastData() {
    }

    // Constructor with all fields
    public ForecastData(String variable, String[] dates, double[] forecast,
                        double[] lower, double[] upper) {
        this.variable = variable;
        this.dates = dates;
        this.forecast = forecast;
        this.lower = lower;
        this.upper = upper;
    }

    // Getters and setters
    public String getVariable() {
        return variable;
    }

    public void setVariable(String variable) {
        this.variable = variable;
    }

    public String[] getDates() {
        return dates;
    }

    public void setDates(String[] dates) {
        this.dates = dates;
    }

    public double[] getForecast() {
        return forecast;
    }

    public void setForecast(double[] forecast) {
        this.forecast = forecast;
    }

    public double[] getLower() {
        return lower;
    }

    public void setLower(double[] lower) {
        this.lower = lower;
    }

    public double[] getUpper() {
        return upper;
    }

    public void setUpper(double[] upper) {
        this.upper = upper;
    }
}
```

**What this does:**
- Represents a forecast for one variable (e.g., inflation predictions)
- `dates` are the forecast periods (e.g., "2025-01", "2025-02")
- `forecast` are the predicted values
- `lower` and `upper` are confidence intervals (e.g., 95% bands)

---

## Step 5: Create the Service Package

```bash
mkdir -p api/src/main/java/com/nip/service
```

Now you'll create a service that reads JSON files from your Python `results/` directory.

---

## Step 6: Create ResultService

**Delete everything in `api/src/main/java/com/nip/service/ResultService.java` and replace it with this:**

```java
package com.nip.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.nip.model.ForecastData;
import com.nip.model.IRFData;
import com.nip.model.VARSummary;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class ResultService {

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final String resultsDir = "results/";

    /**
     * Load VAR summary statistics from JSON file
     */
    public VARSummary getSummary() throws IOException {
        File file = new File(resultsDir + "var_summary.json");
        if (!file.exists()) {
            throw new IOException("var_summary.json not found in results/");
        }
        return objectMapper.readValue(file, VARSummary.class);
    }

    /**
     * Load impulse response function data from JSON file
     */
    public List<IRFData> getIRF() throws IOException {
        File file = new File(resultsDir + "irf_data.json");
        if (!file.exists()) {
            throw new IOException("irf_data.json not found in results/");
        }
        // Assuming the JSON file contains an array of IRF objects
        return objectMapper.readValue(file,
            objectMapper.getTypeFactory().constructCollectionType(List.class, IRFData.class));
    }

    /**
     * Load forecast data from JSON file
     */
    public List<ForecastData> getForecast() throws IOException {
        File file = new File(resultsDir + "forecast_data.json");
        if (!file.exists()) {
            throw new IOException("forecast_data.json not found in results/");
        }
        // Assuming the JSON file contains an array of forecast objects
        return objectMapper.readValue(file,
            objectMapper.getTypeFactory().constructCollectionType(List.class, ForecastData.class));
    }
}
```

**What this does:**
- `@Service` tells Spring Boot this is a service bean (dependency injection)
- `ObjectMapper` is from Jackson library (converts JSON ↔ Java objects)
- `resultsDir` points to the `results/` folder where Python saves JSON files
- Each method reads a JSON file and converts it to your model class
- If the file doesn't exist, it throws an exception

**Important:** This assumes Python has already saved these JSON files:
- `results/var_summary.json`
- `results/irf_data.json`
- `results/forecast_data.json`

---

## Step 7: Create the Controller Package

```bash
mkdir -p api/src/main/java/com/nip/controller
```

Now you'll create the REST controller that exposes your endpoints.

---

## Step 8: Create ResultController

**Delete everything in `api/src/main/java/com/nip/controller/ResultController.java` and replace it with this:**

```java
package com.nip.controller;

import com.nip.model.ForecastData;
import com.nip.model.IRFData;
import com.nip.model.VARSummary;
import com.nip.service.ResultService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/results")
@CrossOrigin(origins = "http://localhost:3000")  // Allow React frontend
public class ResultController {

    @Autowired
    private ResultService resultService;

    /**
     * GET /api/results/summary
     * Returns VAR model summary statistics
     */
    @GetMapping("/summary")
    public ResponseEntity<?> getSummary() {
        try {
            VARSummary summary = resultService.getSummary();
            return ResponseEntity.ok(summary);
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error reading summary data: " + e.getMessage());
        }
    }

    /**
     * GET /api/results/irf
     * Returns impulse response function data
     */
    @GetMapping("/irf")
    public ResponseEntity<?> getIRF() {
        try {
            List<IRFData> irfData = resultService.getIRF();
            return ResponseEntity.ok(irfData);
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error reading IRF data: " + e.getMessage());
        }
    }

    /**
     * GET /api/results/forecast
     * Returns forecast predictions
     */
    @GetMapping("/forecast")
    public ResponseEntity<?> getForecast() {
        try {
            List<ForecastData> forecastData = resultService.getForecast();
            return ResponseEntity.ok(forecastData);
        } catch (IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error reading forecast data: " + e.getMessage());
        }
    }

    /**
     * GET /api/results/health
     * Health check endpoint (for testing)
     */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Result API is running!");
    }
}
```

**What this does:**
- `@RestController` tells Spring Boot this class handles HTTP requests
- `@RequestMapping("/api/results")` sets the base path for all endpoints
- `@CrossOrigin` allows your React frontend (running on port 3000) to call these endpoints
- `@Autowired` injects the `ResultService` (dependency injection)
- Each `@GetMapping` creates an endpoint:
  - `/summary` → returns VAR summary
  - `/irf` → returns IRF data
  - `/forecast` → returns forecast data
  - `/health` → simple test endpoint
- `ResponseEntity<?>` allows you to return different types (success or error)
- `try-catch` handles errors gracefully (returns HTTP 500 if file read fails)

---

## Step 9: Update Application Properties (Optional)

If you want to configure the results directory path, you can add it to your `application.properties`:

**Add this line to `api/src/main/resources/application.properties`:**

```properties
# Results directory
results.directory=results/
```

Then update your `ResultService` to read from this property:

```java
@Value("${results.directory}")
private String resultsDir;
```

For now, we'll keep it simple and hardcode `results/`.

---

## Step 10: Build and Run Your API

Make sure you're in the `api/` directory:

```bash
cd api
./mvnw clean package
java -jar target/nip-api-0.0.1-SNAPSHOT.jar
```

You should see:

```
Started NipApiApplication in 2.345 seconds
Tomcat started on port(s): 8080 (http)
```

---

## Step 11: Test Your Endpoints

Open a new terminal and test each endpoint:

### Test 1: Health Check

```bash
curl http://localhost:8080/api/results/health
```

Expected output:

```
Result API is running!
```

### Test 2: Summary (will fail if JSON doesn't exist yet)

```bash
curl http://localhost:8080/api/results/summary
```

Expected output (if file exists):

```json
{
  "aic": -14.234,
  "bic": -13.876,
  "fpe": 1.23e-07,
  "hqic": -14.089,
  "observations": 120,
  "lags": 3,
  "variables": ["MPR", "TBR", "EXO", "INF"]
}
```

Or (if file doesn't exist):

```
Error reading summary data: var_summary.json not found in results/
```

### Test 3: IRF

```bash
curl http://localhost:8080/api/results/irf
```

### Test 4: Forecast

```bash
curl http://localhost:8080/api/results/forecast
```

You can also open your browser and go to:
- `http://localhost:8080/api/results/health`
- `http://localhost:8080/api/results/summary`

---

## Step 12: Create Mock JSON Files (For Testing)

If you don't have Python results yet, create mock JSON files to test your endpoints:

**Create `results/var_summary.json`:**

```json
{
  "aic": -14.234,
  "bic": -13.876,
  "fpe": 1.23e-07,
  "hqic": -14.089,
  "observations": 120,
  "lags": 3,
  "variables": ["MPR", "TBR", "EXO", "INF"]
}
```

**Create `results/irf_data.json`:**

```json
[
  {
    "shock": "MPR",
    "response": "INF",
    "periods": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "values": [0.0, 0.12, 0.34, 0.52, 0.67, 0.75, 0.79, 0.80, 0.78, 0.74, 0.68, 0.62],
    "lower": [0.0, 0.08, 0.22, 0.35, 0.45, 0.51, 0.53, 0.52, 0.49, 0.44, 0.38, 0.31],
    "upper": [0.0, 0.16, 0.46, 0.69, 0.89, 1.00, 1.05, 1.08, 1.07, 1.04, 0.98, 0.93]
  },
  {
    "shock": "EXO",
    "response": "INF",
    "periods": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "values": [0.0, 0.45, 0.82, 1.05, 1.15, 1.12, 1.01, 0.87, 0.71, 0.55, 0.41, 0.29],
    "lower": [0.0, 0.32, 0.61, 0.78, 0.84, 0.79, 0.67, 0.52, 0.36, 0.20, 0.06, -0.07],
    "upper": [0.0, 0.58, 1.03, 1.32, 1.46, 1.45, 1.35, 1.22, 1.06, 0.90, 0.76, 0.65]
  }
]
```

**Create `results/forecast_data.json`:**

```json
[
  {
    "variable": "INF",
    "dates": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
    "forecast": [21.5, 21.8, 22.1, 22.3, 22.5, 22.6],
    "lower": [20.2, 20.3, 20.4, 20.5, 20.6, 20.7],
    "upper": [22.8, 23.3, 23.8, 24.1, 24.4, 24.5]
  },
  {
    "variable": "MPR",
    "dates": ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"],
    "forecast": [26.25, 26.50, 26.75, 27.00, 27.00, 27.00],
    "lower": [26.25, 26.25, 26.25, 26.50, 26.50, 26.50],
    "upper": [26.25, 26.75, 27.25, 27.50, 27.50, 27.50]
  }
]
```

Now restart your API and test again:

```bash
curl http://localhost:8080/api/results/summary
curl http://localhost:8080/api/results/irf
curl http://localhost:8080/api/results/forecast
```

All three should return JSON data!

---

## Step 13: Commit Your Changes

Stop your API (Ctrl+C), then commit:

```bash
git add api/src/main/java/com/nip/model/
git add api/src/main/java/com/nip/service/
git add api/src/main/java/com/nip/controller/
git add results/var_summary.json
git add results/irf_data.json
git add results/forecast_data.json
git commit -m "Day 32: Add REST endpoints for econometric results

- Created VARSummary, IRFData, ForecastData models
- Created ResultService to read JSON files
- Created ResultController with /summary, /irf, /forecast endpoints
- Added mock JSON files for testing
- All endpoints tested with curl

https://claude.ai/code/session_0173mGraLvrZwx6v1oTi7RnN"
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## Common Errors and Fixes

### Error 1: "Cannot find ObjectMapper"

**Problem:** Missing Jackson dependency.

**Fix:** Add to `pom.xml`:

```xml
<dependency>
    <groupId>com.fasterxml.jackson.core</groupId>
    <artifactId>jackson-databind</artifactId>
</dependency>
```

Then run `./mvnw clean package` again.

### Error 2: "CORS policy: No 'Access-Control-Allow-Origin' header"

**Problem:** Browser blocks requests from React frontend.

**Fix:** Make sure `@CrossOrigin` is in your controller:

```java
@CrossOrigin(origins = "http://localhost:3000")
```

### Error 3: "Error reading summary data: var_summary.json not found"

**Problem:** JSON file doesn't exist in `results/` directory.

**Fix:** Create mock JSON files (see Step 12) or run your Python VAR script to generate real results.

### Error 4: "404 Not Found"

**Problem:** Endpoint URL is wrong.

**Fix:** Check the URL:
- Correct: `http://localhost:8080/api/results/summary`
- Wrong: `http://localhost:8080/results/summary` (missing `/api`)

---

## Questions and Answers

**Q: Why use `ResponseEntity<?>` instead of just returning the object?**

A: `ResponseEntity` gives you control over the HTTP status code. If something goes wrong, you can return a 500 error with a message instead of Spring's default error page.

**Q: What is `@Autowired`?**

A: Spring Boot's dependency injection. Instead of `new ResultService()`, Spring automatically creates one instance and injects it. This makes testing easier and follows best practices.

**Q: Can I test these endpoints without curl?**

A: Yes! Just open your browser and go to `http://localhost:8080/api/results/summary`. You'll see the JSON response. Or use Postman (a GUI tool for testing APIs).

**Q: What if my Python script saves JSON in a different format?**

A: You'll need to update your model classes to match. For example, if Python saves `{"aic_value": -14.23}` instead of `{"aic": -14.23}`, use `@JsonProperty`:

```java
@JsonProperty("aic_value")
private double aic;
```

**Q: Can I add POST endpoints to re-run the VAR model?**

A: Yes! On Day 33, you'll create a POST endpoint that triggers the Python script to re-estimate the model with new data.

---

## What You Accomplished

You now have a fully functional REST API that serves your econometric results! Here's what you built:

1. **Three model classes** — VARSummary, IRFData, ForecastData
2. **A service layer** — ResultService reads JSON files
3. **A controller with 4 endpoints:**
   - `/api/results/health` — health check
   - `/api/results/summary` — VAR summary stats
   - `/api/results/irf` — impulse response functions
   - `/api/results/forecast` — forecast predictions
4. **Error handling** — graceful 500 errors if files are missing
5. **CORS configuration** — allows React frontend to call your API

Tomorrow, you'll create the React frontend that calls these endpoints and displays beautiful charts.

---

## Next Steps (Day 33)

On Day 33, you'll build the React dashboard:
- Create a React app with `create-react-app`
- Fetch data from your REST API using `axios`
- Display VAR summary in a table
- Create line charts for IRF using Recharts library
- Create forecast charts with confidence bands

Your econometric platform is coming together!

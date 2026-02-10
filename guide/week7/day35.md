# Day 35 — API Documentation & Week 7 Wrap-Up

**Goal**: Document your REST API with Swagger/OpenAPI and wrap up Week 7.

**You'll Learn**:
- How to generate interactive API documentation
- How to annotate endpoints for better documentation
- How to access and use Swagger UI
- Week 7 summary and what's next

---

## Why API Documentation Matters

You've built a REST API with multiple endpoints. But if someone else (or future you) wants to use it, they need to know:
- What endpoints exist?
- What parameters do they accept?
- What responses do they return?
- What's the format?

**Swagger/OpenAPI** automatically generates interactive documentation from your code. Users can test endpoints right in the browser.

---

## Step 1: Add springdoc-openapi Dependency

OpenAPI 3 is the modern standard for REST API documentation. `springdoc-openapi` is a Java library that integrates OpenAPI 3 with Spring Boot.

**Action**: Open `pom.xml` and add the springdoc-openapi dependency.

Delete everything in `pom.xml` and replace it with this:

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
        <version>3.2.0</version>
        <relativePath/>
    </parent>

    <groupId>com.inflation</groupId>
    <artifactId>predictor-api</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>Nigerian Inflation Predictor API</name>
    <description>REST API for Nigerian inflation predictions</description>

    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <!-- Spring Boot Web -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Spring Boot JPA -->
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

        <!-- Spring Boot Validation -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-validation</artifactId>
        </dependency>

        <!-- Spring Boot Test -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>

        <!-- SpringDoc OpenAPI UI -->
        <dependency>
            <groupId>org.springdoc</groupId>
            <artifactId>springdoc-openapi-starter-webmvc-ui</artifactId>
            <version>2.3.0</version>
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

**What changed**: Added `springdoc-openapi-starter-webmvc-ui` dependency. This includes:
- OpenAPI 3 specification generator
- Swagger UI (web interface)
- Automatic endpoint discovery

---

## Step 2: Configure OpenAPI in application.properties

Add configuration for API metadata.

**Action**: Open `src/main/resources/application.properties`.

Delete everything in `application.properties` and replace it with this:

```properties
# Database Configuration
spring.datasource.url=jdbc:postgresql://localhost:5432/inflation_db
spring.datasource.username=postgres
spring.datasource.password=yourpassword
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
spring.jpa.properties.hibernate.format_sql=true

# OpenAPI Configuration
springdoc.api-docs.path=/api-docs
springdoc.swagger-ui.path=/swagger-ui.html
springdoc.swagger-ui.operationsSorter=method
springdoc.swagger-ui.tagsSorter=alpha

# API Information
springdoc.info.title=Nigerian Inflation Predictor API
springdoc.info.description=REST API for forecasting Nigerian inflation using VAR models
springdoc.info.version=1.0.0
springdoc.info.contact.name=Your Name
springdoc.info.contact.email=your.email@example.com
```

**What this does**:
- `api-docs.path`: JSON endpoint for OpenAPI specification
- `swagger-ui.path`: Web UI for testing APIs
- `operationsSorter=method`: Sort by HTTP method (GET, POST, etc.)
- `tagsSorter=alpha`: Sort endpoint groups alphabetically
- `info.*`: Metadata shown in Swagger UI

---

## Step 3: Add OpenAPI Annotations to Controllers

Annotations add descriptions, examples, and response codes to your documentation.

**Action**: Open `src/main/java/com/inflation/predictor/controller/ForecastController.java`.

Delete everything in `ForecastController.java` and replace it with this:

```java
package com.inflation.predictor.controller;

import com.inflation.predictor.dto.ForecastRequest;
import com.inflation.predictor.dto.ForecastResponse;
import com.inflation.predictor.service.ForecastService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/forecast")
@Tag(name = "Forecast", description = "Inflation forecasting endpoints")
public class ForecastController {

    @Autowired
    private ForecastService forecastService;

    @Operation(
        summary = "Create a new forecast",
        description = "Generate inflation forecast using VAR model with specified parameters"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Forecast created successfully",
            content = @Content(schema = @Schema(implementation = ForecastResponse.class))
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Invalid input parameters"
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Internal server error during forecast generation"
        )
    })
    @PostMapping
    public ResponseEntity<ForecastResponse> createForecast(
        @Valid @RequestBody ForecastRequest request
    ) {
        ForecastResponse response = forecastService.generateForecast(request);
        return ResponseEntity.ok(response);
    }

    @Operation(
        summary = "Get forecast by ID",
        description = "Retrieve a previously generated forecast by its unique identifier"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Forecast found",
            content = @Content(schema = @Schema(implementation = ForecastResponse.class))
        ),
        @ApiResponse(
            responseCode = "404",
            description = "Forecast not found"
        )
    })
    @GetMapping("/{id}")
    public ResponseEntity<ForecastResponse> getForecast(
        @Parameter(description = "Unique forecast identifier", required = true)
        @PathVariable Long id
    ) {
        ForecastResponse response = forecastService.getForecastById(id);
        return ResponseEntity.ok(response);
    }

    @Operation(
        summary = "Get all forecasts",
        description = "Retrieve list of all forecasts, optionally filtered by scenario"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "List of forecasts retrieved successfully"
        )
    })
    @GetMapping
    public ResponseEntity<?> getAllForecasts(
        @Parameter(description = "Filter by scenario name (optional)")
        @RequestParam(required = false) String scenario
    ) {
        if (scenario != null) {
            return ResponseEntity.ok(forecastService.getForecastsByScenario(scenario));
        }
        return ResponseEntity.ok(forecastService.getAllForecasts());
    }

    @Operation(
        summary = "Delete a forecast",
        description = "Remove a forecast from the database by its ID"
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "204",
            description = "Forecast deleted successfully"
        ),
        @ApiResponse(
            responseCode = "404",
            description = "Forecast not found"
        )
    })
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteForecast(
        @Parameter(description = "Unique forecast identifier", required = true)
        @PathVariable Long id
    ) {
        forecastService.deleteForecast(id);
        return ResponseEntity.noContent().build();
    }
}
```

**Key annotations**:
- `@Tag`: Groups endpoints (shows "Forecast" section in Swagger UI)
- `@Operation`: Describes what an endpoint does
- `@ApiResponses`: Documents possible HTTP status codes
- `@Parameter`: Describes path/query parameters
- `@Schema`: Links to DTO classes for request/response examples

---

## Step 4: Add Annotations to DTOs

Annotate your data transfer objects so Swagger shows field descriptions.

**Action**: Open `src/main/java/com/inflation/predictor/dto/ForecastRequest.java`.

Delete everything in `ForecastRequest.java` and replace it with this:

```java
package com.inflation.predictor.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

@Schema(description = "Request object for generating an inflation forecast")
public class ForecastRequest {

    @Schema(
        description = "Number of periods ahead to forecast",
        example = "12",
        minimum = "1",
        maximum = "24"
    )
    @NotNull(message = "Horizon is required")
    @Min(value = 1, message = "Horizon must be at least 1")
    @Max(value = 24, message = "Horizon cannot exceed 24")
    private Integer horizon;

    @Schema(
        description = "Scenario name for this forecast",
        example = "baseline"
    )
    @NotNull(message = "Scenario is required")
    private String scenario;

    @Schema(
        description = "MPR shock value (percentage points)",
        example = "1.5"
    )
    private Double mprShock;

    @Schema(
        description = "Exchange rate shock value (percentage points)",
        example = "-5.0"
    )
    private Double exoShock;

    // Constructors
    public ForecastRequest() {}

    public ForecastRequest(Integer horizon, String scenario, Double mprShock, Double exoShock) {
        this.horizon = horizon;
        this.scenario = scenario;
        this.mprShock = mprShock;
        this.exoShock = exoShock;
    }

    // Getters and Setters
    public Integer getHorizon() {
        return horizon;
    }

    public void setHorizon(Integer horizon) {
        this.horizon = horizon;
    }

    public String getScenario() {
        return scenario;
    }

    public void setScenario(String scenario) {
        this.scenario = scenario;
    }

    public Double getMprShock() {
        return mprShock;
    }

    public void setMprShock(Double mprShock) {
        this.mprShock = mprShock;
    }

    public Double getExoShock() {
        return exoShock;
    }

    public void setExoShock(Double exoShock) {
        this.exoShock = exoShock;
    }
}
```

**Action**: Now open `src/main/java/com/inflation/predictor/dto/ForecastResponse.java`.

Delete everything in `ForecastResponse.java` and replace it with this:

```java
package com.inflation.predictor.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import java.time.LocalDateTime;
import java.util.List;

@Schema(description = "Response object containing forecast results")
public class ForecastResponse {

    @Schema(description = "Unique forecast identifier", example = "1")
    private Long id;

    @Schema(description = "Scenario name", example = "baseline")
    private String scenario;

    @Schema(description = "Forecast horizon (number of periods)", example = "12")
    private Integer horizon;

    @Schema(description = "Predicted inflation values (one per period)")
    private List<Double> inflationValues;

    @Schema(description = "Confidence interval lower bound")
    private List<Double> lowerBound;

    @Schema(description = "Confidence interval upper bound")
    private List<Double> upperBound;

    @Schema(description = "Timestamp when forecast was created")
    private LocalDateTime createdAt;

    // Constructors
    public ForecastResponse() {}

    public ForecastResponse(Long id, String scenario, Integer horizon,
                           List<Double> inflationValues, List<Double> lowerBound,
                           List<Double> upperBound, LocalDateTime createdAt) {
        this.id = id;
        this.scenario = scenario;
        this.horizon = horizon;
        this.inflationValues = inflationValues;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.createdAt = createdAt;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getScenario() {
        return scenario;
    }

    public void setScenario(String scenario) {
        this.scenario = scenario;
    }

    public Integer getHorizon() {
        return horizon;
    }

    public void setHorizon(Integer horizon) {
        this.horizon = horizon;
    }

    public List<Double> getInflationValues() {
        return inflationValues;
    }

    public void setInflationValues(List<Double> inflationValues) {
        this.inflationValues = inflationValues;
    }

    public List<Double> getLowerBound() {
        return lowerBound;
    }

    public void setLowerBound(List<Double> lowerBound) {
        this.lowerBound = lowerBound;
    }

    public List<Double> getUpperBound() {
        return upperBound;
    }

    public void setUpperBound(List<Double> upperBound) {
        this.upperBound = upperBound;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
}
```

---

## Step 5: Start the Application and Access Swagger UI

Now test your documented API.

**Action**: Start the Spring Boot application.

```bash
cd /home/user/Nigerian_Inflation_Predictor/predictor-api
mvn spring-boot:run
```

Wait for the application to start (look for "Started PredictorApiApplication").

**Action**: Open your browser and navigate to:

```
http://localhost:8080/swagger-ui.html
```

**What you'll see**:
- **Forecast** section with 4 endpoints (POST, GET, GET all, DELETE)
- Click on any endpoint to expand it
- "Try it out" button lets you test right in the browser
- Example request bodies pre-filled from `@Schema` annotations
- Response schemas show expected structure

**Try it**:
1. Click **POST /api/forecast**
2. Click "Try it out"
3. Modify the JSON (or use default)
4. Click "Execute"
5. See response below (status code, body, headers)

---

## Step 6: View the OpenAPI JSON Specification

The raw OpenAPI spec is available as JSON.

**Action**: Navigate to:

```
http://localhost:8080/api-docs
```

You'll see JSON like this:

```json
{
  "openapi": "3.0.1",
  "info": {
    "title": "Nigerian Inflation Predictor API",
    "description": "REST API for forecasting Nigerian inflation using VAR models",
    "contact": {
      "name": "Your Name",
      "email": "your.email@example.com"
    },
    "version": "1.0.0"
  },
  "servers": [
    {
      "url": "http://localhost:8080",
      "description": "Generated server url"
    }
  ],
  "paths": {
    "/api/forecast": { ... },
    "/api/forecast/{id}": { ... }
  }
}
```

This JSON can be imported into tools like Postman, Insomnia, or used to generate client libraries.

---

## Step 7: Create API Reference Documentation

For users who prefer text documentation, create a markdown reference.

**Action**: Create `docs/api_reference.md`.

```bash
mkdir -p /home/user/Nigerian_Inflation_Predictor/docs
```

Delete everything in `docs/api_reference.md` (if it exists) and replace it with this:

```markdown
# API Reference

Base URL: `http://localhost:8080`

---

## Endpoints

### 1. Create Forecast

**POST** `/api/forecast`

Generate a new inflation forecast.

**Request Body**:
```json
{
  "horizon": 12,
  "scenario": "baseline",
  "mprShock": 1.5,
  "exoShock": -5.0
}
```

**Fields**:
- `horizon` (integer, required): Number of periods ahead to forecast (1-24)
- `scenario` (string, required): Scenario name (e.g., "baseline", "optimistic")
- `mprShock` (double, optional): MPR shock value in percentage points
- `exoShock` (double, optional): Exchange rate shock value in percentage points

**Response** (200 OK):
```json
{
  "id": 1,
  "scenario": "baseline",
  "horizon": 12,
  "inflationValues": [15.2, 15.5, 15.8, 16.0, 16.1, 16.2, 16.3, 16.3, 16.2, 16.1, 16.0, 15.9],
  "lowerBound": [14.5, 14.7, 14.9, 15.0, 15.1, 15.1, 15.2, 15.2, 15.1, 15.0, 14.9, 14.8],
  "upperBound": [15.9, 16.3, 16.7, 17.0, 17.1, 17.3, 17.4, 17.4, 17.3, 17.2, 17.1, 17.0],
  "createdAt": "2026-02-10T14:30:00"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input parameters
- `500 Internal Server Error`: Forecast generation failed

---

### 2. Get Forecast by ID

**GET** `/api/forecast/{id}`

Retrieve a specific forecast.

**Path Parameters**:
- `id` (long): Unique forecast identifier

**Response** (200 OK):
Same structure as POST response.

**Error Responses**:
- `404 Not Found`: Forecast with given ID does not exist

---

### 3. Get All Forecasts

**GET** `/api/forecast`

Retrieve all forecasts, optionally filtered by scenario.

**Query Parameters**:
- `scenario` (string, optional): Filter by scenario name

**Examples**:
- Get all: `GET /api/forecast`
- Filter by scenario: `GET /api/forecast?scenario=baseline`

**Response** (200 OK):
Array of forecast objects.

---

### 4. Delete Forecast

**DELETE** `/api/forecast/{id}`

Remove a forecast from the database.

**Path Parameters**:
- `id` (long): Unique forecast identifier

**Response** (204 No Content):
Empty body on success.

**Error Responses**:
- `404 Not Found`: Forecast with given ID does not exist

---

## Authentication

Currently, no authentication is required. For production deployment, implement JWT or OAuth2.

---

## Rate Limiting

No rate limiting is currently enforced. Consider adding for production.

---

## Versioning

API version: `v1.0.0`

Future versions may use URL versioning (e.g., `/api/v2/forecast`).
```

---

## Step 8: Commit Your Changes

**Action**: Stop the application (Ctrl+C), then commit.

```bash
cd /home/user/Nigerian_Inflation_Predictor
git add predictor-api/pom.xml
git add predictor-api/src/main/resources/application.properties
git add predictor-api/src/main/java/com/inflation/predictor/controller/ForecastController.java
git add predictor-api/src/main/java/com/inflation/predictor/dto/ForecastRequest.java
git add predictor-api/src/main/java/com/inflation/predictor/dto/ForecastResponse.java
git add docs/api_reference.md
git commit -m "Day 35: Add OpenAPI documentation with Swagger UI"
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## Week 7 Recap

You built a complete Spring Boot REST API over 5 days:

| Day | Topic | What You Built |
|-----|-------|----------------|
| 31 | Spring Boot Setup | Maven project, controller, first endpoint |
| 32 | Database Integration | PostgreSQL connection, JPA entities, repository |
| 33 | Service Layer & DTOs | Business logic, validation, request/response objects |
| 34 | Testing | Unit tests (MockMvc), integration tests (TestRestTemplate) |
| 35 | Documentation | OpenAPI annotations, Swagger UI, API reference |

**Key Concepts Learned**:
- **REST principles**: GET, POST, PUT, DELETE
- **Spring Boot annotations**: @RestController, @Service, @Repository, @Entity
- **Dependency Injection**: @Autowired
- **JPA**: Object-relational mapping, saving/retrieving data
- **Validation**: @Valid, @NotNull, @Min, @Max
- **Testing**: Mocking services, testing endpoints
- **Documentation**: OpenAPI 3, Swagger UI

**Architecture**:
```
Controller (HTTP) → Service (business logic) → Repository (database) → PostgreSQL
```

---

## What's Next: Week 8 (Days 36-40)

Week 8 will focus on **deployment and production readiness**:

- **Day 36**: Dockerizing the Spring Boot application
- **Day 37**: Environment-specific configurations (dev, staging, prod)
- **Day 38**: Logging and monitoring with Spring Boot Actuator
- **Day 39**: Security basics (CORS, JWT authentication)
- **Day 40**: Cloud deployment (Heroku or AWS) & final recap

By the end of Week 8, you'll have a production-ready, deployed API.

---

## Common Errors & Fixes

### Error: "Failed to start bean 'documentationPluginsBootstrapper'"

**Cause**: Wrong Swagger library version or Spring Boot incompatibility.

**Fix**: Use `springdoc-openapi-starter-webmvc-ui` version 2.3.0 (as shown in Step 1). Older `springfox` libraries don't work with Spring Boot 3.x.

---

### Error: Swagger UI shows no endpoints

**Cause**: Controllers not detected or wrong package scanning.

**Fix**: Ensure your controllers are in `com.inflation.predictor.controller` (under the main application package). Spring Boot auto-scans this structure.

---

### Error: Example values not showing in Swagger UI

**Cause**: Missing `@Schema` annotations.

**Fix**: Add `@Schema(example = "...")` to DTO fields (as shown in Step 4).

---

### Error: "Content type 'application/xml' not supported"

**Cause**: You're sending XML instead of JSON.

**Fix**: In Swagger UI, ensure "Content-Type" header is `application/json`. In Postman/Insomnia, select "JSON" body type.

---

## Questions & Answers

**Q1: Do I need to manually update Swagger UI when I change my code?**

No. Swagger UI is generated automatically from your code. Every time you restart the application, it reflects your latest endpoints and annotations.

---

**Q2: Can I hide certain endpoints from Swagger UI?**

Yes. Add `@Hidden` annotation from `io.swagger.v3.oas.annotations` to a controller or method:

```java
@Hidden
@GetMapping("/internal")
public String internalEndpoint() { ... }
```

---

**Q3: How do I add authentication to Swagger UI?**

Add a security scheme configuration class:

```java
@Configuration
public class OpenAPIConfig {
    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
            .components(new Components()
                .addSecuritySchemes("bearer-jwt",
                    new SecurityScheme()
                        .type(SecurityScheme.Type.HTTP)
                        .scheme("bearer")
                        .bearerFormat("JWT")));
    }
}
```

Then annotate your controllers with `@SecurityRequirement(name = "bearer-jwt")`.

---

**Q4: Can I export the OpenAPI spec to a file?**

Yes. Add this to `application.properties`:

```properties
springdoc.api-docs.path=/api-docs
springdoc.writer.with-default-pretty-printer=true
```

Then visit `http://localhost:8080/api-docs` and save the JSON. You can import this into Postman or generate client libraries with tools like OpenAPI Generator.

---

**Q5: What's the difference between OpenAPI and Swagger?**

- **OpenAPI**: The specification standard (like a blueprint)
- **Swagger**: A set of tools for working with OpenAPI (Swagger UI is one tool)

Think of OpenAPI as the language, Swagger as the dictionary.

---

**Q6: Do I need Swagger UI in production?**

It depends. For public APIs, yes (helps users understand your API). For internal-only APIs, you might disable it in production for security. Add this to production config:

```properties
springdoc.swagger-ui.enabled=false
```

---

**Q7: Can I customize the Swagger UI theme?**

Yes, but it requires custom CSS. You can change colors, fonts, etc. by adding a custom CSS file and referencing it in `application.properties`:

```properties
springdoc.swagger-ui.custom-css=/custom-swagger.css
```

---

## Summary

Today you:
1. ✅ Added springdoc-openapi dependency
2. ✅ Configured OpenAPI in application.properties
3. ✅ Annotated controllers and DTOs for documentation
4. ✅ Tested endpoints in Swagger UI
5. ✅ Created a markdown API reference
6. ✅ Committed your changes

**Week 7 Complete**: You have a fully functional, tested, and documented REST API.

**Next**: Day 36 — Dockerize your application for easy deployment.

# Day 34 — API Testing & Error Handling

## What You'll Learn Today

Today you'll make your Spring Boot API production-ready by adding:
- **Integration tests** that verify your endpoints work correctly
- **Error handling** so users get helpful messages instead of cryptic stack traces
- **CORS configuration** so your frontend can talk to your backend

By the end, you'll have a tested, robust API with proper error responses.

---

## Why This Matters

Right now, if something goes wrong, Spring Boot shows a generic error page. Users see confusing messages like "Whitelabel Error Page." That's not professional.

Also, without tests, you don't know if your code works until you manually test it in a browser. That doesn't scale.

Today you automate testing and improve error messages.

---

## Step 1: Create the Test Class

Spring Boot has amazing testing support. You'll use **MockMvc** to test your endpoints without starting a real server.

Create this file:

**Delete everything in `src/test/java/com/inflation/predictor/InflationControllerTest.java` and replace it with this:**

```java
package com.inflation.predictor;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.hamcrest.Matchers.*;

@SpringBootTest
@AutoConfigureMockMvc
public class InflationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    public void testGetForecast() throws Exception {
        mockMvc.perform(get("/api/forecast")
                .param("horizon", "12"))
                .andExpect(status().isOk())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.forecast", hasSize(12)))
                .andExpect(jsonPath("$.forecast[0].month", notNullValue()))
                .andExpect(jsonPath("$.forecast[0].predicted_inflation", notNullValue()));
    }

    @Test
    public void testGetHistorical() throws Exception {
        mockMvc.perform(get("/api/historical"))
                .andExpect(status().isOk())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.data", isA(java.util.List.class)))
                .andExpect(jsonPath("$.data", hasSize(greaterThan(0))))
                .andExpect(jsonPath("$.data[0].date", notNullValue()))
                .andExpect(jsonPath("$.data[0].infl", notNullValue()));
    }

    @Test
    public void testGetImpulseResponse() throws Exception {
        mockMvc.perform(get("/api/impulse-response")
                .param("shock", "mpr")
                .param("periods", "24"))
                .andExpect(status().isOk())
                .andExpect(content().contentType(MediaType.APPLICATION_JSON))
                .andExpect(jsonPath("$.shock", is("mpr")))
                .andExpect(jsonPath("$.responses", hasSize(24)))
                .andExpect(jsonPath("$.responses[0].period", notNullValue()))
                .andExpect(jsonPath("$.responses[0].impact_on_inflation", notNullValue()));
    }

    @Test
    public void testInvalidHorizon() throws Exception {
        mockMvc.perform(get("/api/forecast")
                .param("horizon", "100"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", notNullValue()))
                .andExpect(jsonPath("$.message", containsString("Horizon")));
    }

    @Test
    public void testInvalidShock() throws Exception {
        mockMvc.perform(get("/api/impulse-response")
                .param("shock", "invalid_variable")
                .param("periods", "12"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", notNullValue()))
                .andExpect(jsonPath("$.message", containsString("shock")));
    }

    @Test
    public void testMissingParameter() throws Exception {
        mockMvc.perform(get("/api/forecast"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", notNullValue()));
    }
}
```

**What this does:**
- `@SpringBootTest`: Loads the full Spring context (your app)
- `@AutoConfigureMockMvc`: Gives you a MockMvc object to test endpoints
- `mockMvc.perform(get(...))`: Makes a GET request
- `.andExpect(status().isOk())`: Checks HTTP 200
- `.andExpect(jsonPath(...))`: Checks JSON structure

You have **6 tests**:
1. Forecast endpoint returns correct JSON
2. Historical data endpoint works
3. Impulse response endpoint works
4. Invalid horizon returns 400 Bad Request
5. Invalid shock variable returns 400
6. Missing parameter returns 400

**Note:** For these tests to pass, you need error handling. Let's add it.

---

## Step 2: Add Global Error Handling

Create an exception handler that catches errors and returns nice JSON responses.

**Delete everything in `src/main/java/com/inflation/predictor/GlobalExceptionHandler.java` and replace it with this:**

```java
package com.inflation.predictor;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> handleIllegalArgument(IllegalArgumentException ex) {
        Map<String, Object> body = new HashMap<>();
        body.put("timestamp", LocalDateTime.now());
        body.put("status", HttpStatus.BAD_REQUEST.value());
        body.put("error", "Bad Request");
        body.put("message", ex.getMessage());

        return new ResponseEntity<>(body, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<Map<String, Object>> handleMissingParams(MissingServletRequestParameterException ex) {
        Map<String, Object> body = new HashMap<>();
        body.put("timestamp", LocalDateTime.now());
        body.put("status", HttpStatus.BAD_REQUEST.value());
        body.put("error", "Bad Request");
        body.put("message", "Missing required parameter: " + ex.getParameterName());

        return new ResponseEntity<>(body, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<Map<String, Object>> handleTypeMismatch(MethodArgumentTypeMismatchException ex) {
        Map<String, Object> body = new HashMap<>();
        body.put("timestamp", LocalDateTime.now());
        body.put("status", HttpStatus.BAD_REQUEST.value());
        body.put("error", "Bad Request");
        body.put("message", "Invalid value for parameter: " + ex.getName());

        return new ResponseEntity<>(body, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> handleGenericException(Exception ex) {
        Map<String, Object> body = new HashMap<>();
        body.put("timestamp", LocalDateTime.now());
        body.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
        body.put("error", "Internal Server Error");
        body.put("message", "An unexpected error occurred");

        return new ResponseEntity<>(body, HttpStatus.INTERNAL_SERVER_ERROR);
    }
}
```

**What this does:**
- `@ControllerAdvice`: This class handles exceptions globally across all controllers
- `@ExceptionHandler`: Catches specific exception types
- Returns a consistent JSON structure with timestamp, status, error, and message
- Different handlers for different error types

Now update your controller to throw proper exceptions.

---

## Step 3: Update Controller with Validation

**Delete everything in `src/main/java/com/inflation/predictor/InflationController.java` and replace it with this:**

```java
package com.inflation.predictor;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.*;

@RestController
@RequestMapping("/api")
public class InflationController {

    @Autowired
    private InflationService inflationService;

    @GetMapping("/forecast")
    public Map<String, Object> getForecast(@RequestParam int horizon) {
        if (horizon < 1 || horizon > 36) {
            throw new IllegalArgumentException("Horizon must be between 1 and 36 months");
        }

        List<Map<String, Object>> forecast = inflationService.generateForecast(horizon);

        Map<String, Object> response = new HashMap<>();
        response.put("forecast", forecast);
        response.put("horizon", horizon);
        response.put("generated_at", new Date());

        return response;
    }

    @GetMapping("/historical")
    public Map<String, Object> getHistorical() {
        List<Map<String, Object>> data = inflationService.getHistoricalData();

        Map<String, Object> response = new HashMap<>();
        response.put("data", data);
        response.put("count", data.size());

        return response;
    }

    @GetMapping("/impulse-response")
    public Map<String, Object> getImpulseResponse(
            @RequestParam String shock,
            @RequestParam(defaultValue = "24") int periods) {

        // Validate shock variable
        List<String> validShocks = Arrays.asList("mpr", "tbr", "exo");
        if (!validShocks.contains(shock.toLowerCase())) {
            throw new IllegalArgumentException(
                "Invalid shock variable. Must be one of: mpr, tbr, exo");
        }

        if (periods < 1 || periods > 60) {
            throw new IllegalArgumentException("Periods must be between 1 and 60");
        }

        List<Map<String, Object>> responses = inflationService.calculateImpulseResponse(shock, periods);

        Map<String, Object> response = new HashMap<>();
        response.put("shock", shock);
        response.put("periods", periods);
        response.put("responses", responses);

        return response;
    }

    @GetMapping("/variance-decomposition")
    public Map<String, Object> getVarianceDecomposition(@RequestParam(defaultValue = "12") int periods) {
        if (periods < 1 || periods > 60) {
            throw new IllegalArgumentException("Periods must be between 1 and 60");
        }

        Map<String, Object> decomposition = inflationService.calculateVarianceDecomposition(periods);

        Map<String, Object> response = new HashMap<>();
        response.put("periods", periods);
        response.put("decomposition", decomposition);

        return response;
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        Map<String, String> status = new HashMap<>();
        status.put("status", "UP");
        status.put("timestamp", new Date().toString());
        return status;
    }
}
```

**Changes:**
- Added validation that throws `IllegalArgumentException` with helpful messages
- Horizon must be 1-36 months (reasonable forecast range)
- Shock variable must be mpr, tbr, or exo
- Periods must be 1-60
- Added `/health` endpoint for monitoring

---

## Step 4: Add CORS Configuration

If your frontend runs on a different port (e.g., React on port 3000, Spring on 8080), you need **CORS** (Cross-Origin Resource Sharing).

**Delete everything in `src/main/java/com/inflation/predictor/WebConfig.java` and replace it with this:**

```java
package com.inflation.predictor;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins(
                    "http://localhost:3000",
                    "http://localhost:5173",
                    "https://your-production-domain.com"
                )
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .allowCredentials(true);
    }
}
```

**What this does:**
- Allows requests from port 3000 (React/Next.js) and 5173 (Vite)
- Allows all HTTP methods
- Allows credentials (cookies, auth headers)
- Only applies to `/api/**` endpoints

**In production**, replace `http://localhost:3000` with your actual frontend URL.

---

## Step 5: Run the Tests

Now run your tests:

```bash
cd /home/user/Nigerian_Inflation_Predictor
mvn test
```

**Expected output:**

```
[INFO] -------------------------------------------------------
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running com.inflation.predictor.InflationControllerTest
[INFO] Tests run: 6, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 2.341 s
[INFO]
[INFO] Results:
[INFO]
[INFO] Tests run: 6, Failures: 0, Errors: 0, Skipped: 0
[INFO]
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
```

All 6 tests should pass.

---

## Step 6: Test Error Handling Manually

Start your server:

```bash
mvn spring-boot:run
```

Test invalid requests:

**Test 1: Invalid horizon**
```bash
curl "http://localhost:8080/api/forecast?horizon=100"
```

Response:
```json
{
  "timestamp": "2026-02-10T14:35:22",
  "status": 400,
  "error": "Bad Request",
  "message": "Horizon must be between 1 and 36 months"
}
```

**Test 2: Missing parameter**
```bash
curl "http://localhost:8080/api/forecast"
```

Response:
```json
{
  "timestamp": "2026-02-10T14:36:15",
  "status": 400,
  "error": "Bad Request",
  "message": "Missing required parameter: horizon"
}
```

**Test 3: Invalid shock**
```bash
curl "http://localhost:8080/api/impulse-response?shock=gdp&periods=12"
```

Response:
```json
{
  "timestamp": "2026-02-10T14:37:08",
  "status": 400,
  "error": "Bad Request",
  "message": "Invalid shock variable. Must be one of: mpr, tbr, exo"
}
```

Beautiful! Users now see helpful error messages instead of stack traces.

---

## Step 7: Commit Your Changes

```bash
git add src/test/java/com/inflation/predictor/InflationControllerTest.java
git add src/main/java/com/inflation/predictor/GlobalExceptionHandler.java
git add src/main/java/com/inflation/predictor/InflationController.java
git add src/main/java/com/inflation/predictor/WebConfig.java
git commit -m "Add integration tests, error handling, and CORS configuration"
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## Common Errors

### Error 1: Tests fail with "404 Not Found"

**Problem:** Your endpoints don't match the test URLs.

**Solution:** Check that your controller has `@RequestMapping("/api")` and methods have the correct paths.

---

### Error 2: "NoSuchBeanDefinitionException: No qualifying bean of type 'InflationService'"

**Problem:** Your service isn't being detected by Spring.

**Solution:** Make sure `InflationService` has `@Service` annotation:

```java
@Service
public class InflationService {
    // ...
}
```

---

### Error 3: Tests fail with "jsonPath" errors

**Problem:** The JSON structure doesn't match what you're testing.

**Solution:** Run your server manually and check the actual JSON response:

```bash
curl http://localhost:8080/api/forecast?horizon=12 | jq
```

Update your test expectations to match the actual structure.

---

### Error 4: CORS errors in browser console

**Problem:** Your frontend can't access the API.

**Solution:**
1. Check `WebConfig.java` includes your frontend's URL
2. Make sure you restart Spring Boot after adding CORS config
3. Check browser console for the exact error message

---

## Questions & Answers

**Q: Why use `@ControllerAdvice` instead of try-catch in every method?**

A: `@ControllerAdvice` is **global**. You write error handling once, it applies everywhere. Without it, you'd need try-catch in every single controller method. That's repetitive and error-prone.

---

**Q: What's the difference between unit tests and integration tests?**

A:
- **Unit tests**: Test one class in isolation (mock everything else)
- **Integration tests**: Test multiple components together (your controller + service + database)

Your tests today are **integration tests** because they test the full stack with `@SpringBootTest`.

---

**Q: Should I test every possible error case?**

A: Test the **most likely** error cases:
- Missing parameters (users forget them)
- Invalid values (users type wrong things)
- Boundary conditions (negative numbers, huge numbers)

Don't test every possible exception. Focus on what users will actually encounter.

---

**Q: Why return HTTP 400 vs 500?**

A:
- **400 Bad Request**: User's fault (invalid input)
- **500 Internal Server Error**: Your fault (bug in code)

Always use 400 for validation errors. Users can fix those. Use 500 only for unexpected crashes.

---

**Q: Can I customize the error JSON structure?**

A: Yes! Just change the `Map` you return in `GlobalExceptionHandler`. You could add:
- `path`: Which endpoint failed
- `details`: More specific error info
- `code`: Custom error codes

Example:
```java
body.put("path", request.getRequestURI());
body.put("code", "INVALID_HORIZON");
```

---

**Q: How do I test authentication/authorization?**

A: You'll add that in Week 8. For now, your API is public (no auth). Later you'll use:
- Spring Security
- JWT tokens
- `@WithMockUser` in tests

---

## What You Learned

1. **Integration testing** with MockMvc lets you test endpoints without manual clicking
2. **@ControllerAdvice** centralizes error handling across all controllers
3. **CORS configuration** allows frontend/backend on different ports
4. **Validation** in controllers prevents bad data from reaching your service layer
5. **Consistent error responses** make your API professional and debuggable

---

## Tomorrow: Day 35

Tomorrow you'll add **caching** to improve performance. Right now, every API call recalculates the forecast. That's slow. You'll use Spring Cache to store results and serve them instantly on repeated requests.

You'll learn:
- How to add Redis caching
- Cache invalidation strategies
- Performance benchmarking before/after caching

See you tomorrow!

# Day 31 — Spring Boot API: Project Setup

## What You'll Learn Today
- What a REST API is and why you need one for your inflation predictor
- How to set up a Java 17 + Spring Boot 3 + Maven project
- How to create the project skeleton and verify it runs
- The role of each configuration file in a Spring Boot application

By the end of today, you'll have a working Spring Boot application that starts successfully and responds to HTTP requests.

---

## Theory: Why an API?

### The Problem
Your Python scripts produce results:
- JSON files with VAR forecasts
- CSV tables with impulse response functions
- Cointegration test outputs

But these files sit in your `data/results/` folder. How does a web browser, mobile app, or another system access them?

### The Solution: A REST API
A **REST API** (Representational State Transfer Application Programming Interface) is a web service that:
1. Listens for HTTP requests (like when you visit a website)
2. Fetches data from a database or file system
3. Sends back data as JSON

**Example:**
```
GET http://localhost:8080/api/forecasts
→ Returns JSON: {"inflation_forecast": [12.5, 13.1, 13.7], "horizon": 3}
```

### Why Spring Boot?
**Spring Boot** is a Java framework that makes building REST APIs easy:
- Auto-configures web servers, database connections, and JSON serialization
- Production-ready (used by banks, airlines, Netflix)
- Strong ecosystem for security, monitoring, testing

### Architecture
```
[Python Scripts] → [PostgreSQL Database] → [Spring Boot API] → [React Dashboard]
     (compute)         (store results)        (serve data)        (display)
```

Your Python layer does the heavy econometric computation. The Java API layer serves the results efficiently.

---

## Prerequisites

### 1. Install Java 17
Spring Boot 3 requires Java 17 or higher.

**Check if you already have it:**
```bash
java -version
```

**If you see "openjdk 17" or "java 17", you're good. If not:**

- **Ubuntu/Debian:**
  ```bash
  sudo apt update
  sudo apt install openjdk-17-jdk
  ```

- **macOS (with Homebrew):**
  ```bash
  brew install openjdk@17
  ```

- **Windows:**
  Download from [https://adoptium.net/](https://adoptium.net/) and install.

**Verify:**
```bash
java -version
# Should show: openjdk version "17.x.x"
```

### 2. Install Maven
**Maven** is a build tool for Java projects (like `pip` for Python, but also handles compilation).

**Check if you already have it:**
```bash
mvn -version
```

**If not installed:**

- **Ubuntu/Debian:**
  ```bash
  sudo apt install maven
  ```

- **macOS:**
  ```bash
  brew install maven
  ```

- **Windows:**
  Download from [https://maven.apache.org/download.cgi](https://maven.apache.org/download.cgi), extract, and add to PATH.

**Verify:**
```bash
mvn -version
# Should show: Apache Maven 3.x.x
```

---

## Step 1: Create Project Structure

### Option A: Using Spring Initializr (Recommended for Beginners)

1. **Go to [https://start.spring.io/](https://start.spring.io/)**

2. **Configure your project:**
   - **Project:** Maven
   - **Language:** Java
   - **Spring Boot:** 3.2.2 (or latest 3.x.x)
   - **Group:** `com.cbn`
   - **Artifact:** `inflation-api`
   - **Name:** `inflation-api`
   - **Package name:** `com.cbn.inflationapi`
   - **Packaging:** Jar
   - **Java:** 17

3. **Add dependencies** (click "ADD DEPENDENCIES"):
   - Spring Web
   - Spring Data JPA
   - PostgreSQL Driver

4. **Click "GENERATE"** — downloads a ZIP file

5. **Extract the ZIP into your project:**
   ```bash
   cd /home/user/Nigerian_Inflation_Predictor
   unzip ~/Downloads/inflation-api.zip -d api/
   ```

Now you have a folder `api/inflation-api/` with the project skeleton.

### Option B: Manual Setup (We'll Do This)

We'll create the project manually so you understand every file.

**Create the directory structure:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
mkdir -p api/inflation-api/src/main/java/com/cbn/inflationapi
mkdir -p api/inflation-api/src/main/resources
mkdir -p api/inflation-api/src/test/java/com/cbn/inflationapi
```

**What each folder means:**
- `src/main/java/` — Your Java source code
- `src/main/resources/` — Configuration files (properties, SQL scripts)
- `src/test/java/` — Unit tests (we'll add later)
- `com/cbn/inflationapi/` — Java package structure (company/project/app)

---

## Step 2: Create pom.xml

The **pom.xml** (Project Object Model) tells Maven:
- What dependencies to download (Spring Boot, PostgreSQL driver)
- How to build your project
- What Java version to use

**Delete everything in `api/inflation-api/pom.xml` and replace it with this:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <!-- Spring Boot Parent POM (provides default configs) -->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.2</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <!-- Your project coordinates -->
    <groupId>com.cbn</groupId>
    <artifactId>inflation-api</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>inflation-api</name>
    <description>REST API for Nigerian Inflation Predictor</description>

    <!-- Java version -->
    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <!-- Spring Boot Web: REST API, embedded Tomcat server -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Spring Boot Data JPA: Database access with Hibernate -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>

        <!-- PostgreSQL Driver: Connects to your database -->
        <dependency>
            <groupId>org.postgresql</groupId>
            <artifactId>postgresql</artifactId>
            <scope>runtime</scope>
        </dependency>

        <!-- Jackson Databind: JSON serialization/deserialization -->
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
        </dependency>

        <!-- Spring Boot DevTools: Auto-restart on code changes (optional) -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>

        <!-- Spring Boot Test: Unit testing framework -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Spring Boot Maven Plugin: Packages app as executable JAR -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>
```

**What each dependency does:**
- **spring-boot-starter-web:** Includes Spring MVC, REST support, and embedded Tomcat server
- **spring-boot-starter-data-jpa:** Hibernate ORM for database operations
- **postgresql:** JDBC driver to connect to PostgreSQL
- **jackson-databind:** Converts Java objects ↔ JSON automatically
- **spring-boot-devtools:** Restarts app when you change code (saves time during development)
- **spring-boot-starter-test:** JUnit, Mockito for testing

**Save the file.**

---

## Step 3: Create the Main Application Class

Every Spring Boot application needs a main class with `@SpringBootApplication` annotation. This is the entry point.

**Delete everything in `api/inflation-api/src/main/java/com/cbn/inflationapi/InflationApiApplication.java` and replace it with this:**

```java
package com.cbn.inflationapi;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main entry point for the Inflation API.
 *
 * The @SpringBootApplication annotation:
 * - Enables auto-configuration (Spring Boot guesses what you need)
 * - Enables component scanning (finds your @Controller, @Service classes)
 * - Marks this as a configuration class
 */
@SpringBootApplication
public class InflationApiApplication {

    public static void main(String[] args) {
        // Starts the embedded Tomcat server and initializes Spring context
        SpringApplication.run(InflationApiApplication.class, args);
    }

}
```

**Explanation:**
- **`@SpringBootApplication`**: Combines three annotations:
  - `@Configuration`: Declares this class as a source of bean definitions
  - `@EnableAutoConfiguration`: Tells Spring Boot to auto-configure based on your dependencies
  - `@ComponentScan`: Scans `com.cbn.inflationapi` package for components

- **`SpringApplication.run(...)`**: Starts the web server and loads your application context

**Why Java is verbose:** Java requires explicit package declarations, imports, and type annotations. Python is more concise, but Java's verbosity helps catch errors at compile time.

**Save the file.**

---

## Step 4: Create application.properties

This file configures your application: server port, database connection, logging.

**Delete everything in `api/inflation-api/src/main/resources/application.properties` and replace it with this:**

```properties
# Application Name
spring.application.name=inflation-api

# Server Configuration
server.port=8080
server.servlet.context-path=/

# PostgreSQL Database Configuration (we'll connect to DB in Day 32)
# For now, we'll disable JPA auto-configuration to avoid errors
spring.datasource.url=jdbc:postgresql://localhost:5432/cbn_inflation
spring.datasource.username=postgres
spring.datasource.password=your_password_here
spring.jpa.hibernate.ddl-auto=none
spring.jpa.show-sql=false

# Temporarily disable DataSource auto-configuration (no DB needed yet)
spring.autoconfigure.exclude=org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration

# Jackson JSON Configuration
spring.jackson.serialization.write-dates-as-timestamps=false
spring.jackson.time-zone=Africa/Lagos

# Logging
logging.level.root=INFO
logging.level.com.cbn.inflationapi=DEBUG
logging.pattern.console=%d{yyyy-MM-dd HH:mm:ss} - %msg%n
```

**Explanation:**
- **`server.port=8080`**: Your API will run on `http://localhost:8080`
- **`spring.datasource.*`**: Database credentials (we'll use this in Day 32)
- **`spring.autoconfigure.exclude`**: Disables database auto-configuration for now (we haven't set up PostgreSQL yet). This prevents "Cannot load driver class" errors.
- **`spring.jackson.*`**: Configures JSON serialization (dates as ISO strings, not timestamps)
- **`logging.level.*`**: Sets log verbosity (INFO for general logs, DEBUG for your app)

**Important:** Replace `your_password_here` with your actual PostgreSQL password (if you have one set up). For now, this doesn't matter because we're excluding DataSource auto-configuration.

**Save the file.**

---

## Step 5: Verify the Setup

### Download Dependencies
First, let Maven download all dependencies (Spring Boot, PostgreSQL driver, etc.):

```bash
cd /home/user/Nigerian_Inflation_Predictor/api/inflation-api
mvn clean install
```

**What this does:**
- **`clean`**: Deletes old build files
- **`install`**: Downloads dependencies, compiles code, runs tests, packages as JAR

**Expected output:**
```
[INFO] BUILD SUCCESS
[INFO] Total time: 45.123 s
```

**If you see errors about Java version:**
```
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
mvn clean install
```

### Run the Application
```bash
mvn spring-boot:run
```

**Expected output:**
```
  .   ____          _            __ _ _
 /\\ / ___'_ __ _ _(_)_ __  __ _ \ \ \ \
( ( )\___ | '_ | '_| | '_ \/ _` | \ \ \ \
 \\/  ___)| |_)| | | | | || (_| |  ) ) ) )
  '  |____| .__|_| |_|_| |_\__, | / / / /
 =========|_|==============|___/=/_/_/_/
 :: Spring Boot ::                (v3.2.2)

2026-02-10 14:23:45 - Starting InflationApiApplication...
2026-02-10 14:23:47 - Tomcat started on port(s): 8080 (http)
2026-02-10 14:23:47 - Started InflationApiApplication in 2.543 seconds
```

**Success!** Your Spring Boot application is running.

### Test the Application
Open a new terminal (keep the server running) and test:

```bash
curl http://localhost:8080
```

**Expected output:**
```json
{"timestamp":"2026-02-10T14:25:00.000+00:00","status":404,"error":"Not Found","path":"/"}
```

**This is GOOD!** A 404 error means the server is responding. We haven't created any endpoints yet, so it doesn't know what to serve at `/`. We'll fix this in Day 32.

**Stop the server:** Press `Ctrl+C` in the terminal where `mvn spring-boot:run` is running.

---

## Step 6: Create a Test Endpoint (Optional)

Let's create a simple "Hello World" endpoint to verify the API works.

**Create a new file `api/inflation-api/src/main/java/com/cbn/inflationapi/HealthController.java` and add this:**

```java
package com.cbn.inflationapi;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

/**
 * Health check endpoint to verify the API is running.
 */
@RestController
public class HealthController {

    @GetMapping("/health")
    public Map<String, String> health() {
        Map<String, String> response = new HashMap<>();
        response.put("status", "UP");
        response.put("message", "Inflation API is running");
        return response;
    }

}
```

**Explanation:**
- **`@RestController`**: Marks this class as a REST controller (Spring will scan and register it)
- **`@GetMapping("/health")`**: Maps HTTP GET requests to `/health` to this method
- **Return `Map<String, String>`**: Spring Boot automatically converts this to JSON

**Restart the application:**
```bash
mvn spring-boot:run
```

**Test the new endpoint:**
```bash
curl http://localhost:8080/health
```

**Expected output:**
```json
{"status":"UP","message":"Inflation API is running"}
```

**Perfect!** You've created your first REST endpoint.

---

## Git Commit

Let's save your work.

**Add the new files:**
```bash
cd /home/user/Nigerian_Inflation_Predictor
git add api/inflation-api/pom.xml
git add api/inflation-api/src/main/java/com/cbn/inflationapi/InflationApiApplication.java
git add api/inflation-api/src/main/resources/application.properties
git add api/inflation-api/src/main/java/com/cbn/inflationapi/HealthController.java
```

**Commit:**
```bash
git commit -m "Day 31: Set up Spring Boot API project

- Created Maven project structure with pom.xml
- Added dependencies: Spring Web, JPA, PostgreSQL, Jackson
- Created main application class (InflationApiApplication)
- Configured application.properties (server port, logging)
- Added /health endpoint for testing
- Verified app starts successfully on port 8080"
```

**Push to remote:**
```bash
git push origin claude/inflation-predictor-platform-JG3LF
```

---

## Common Errors and Fixes

### Error: "Failed to configure a DataSource"
**Cause:** Spring Boot tries to connect to a database, but you haven't set one up yet.

**Fix:** Make sure `application.properties` has this line:
```properties
spring.autoconfigure.exclude=org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration
```

This disables database auto-configuration until we set up PostgreSQL in Day 32.

### Error: "java.lang.UnsupportedClassVersionError"
**Cause:** You're running Java 11 or older, but Spring Boot 3 requires Java 17.

**Fix:** Install Java 17:
```bash
sudo apt install openjdk-17-jdk
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
```

### Error: "Port 8080 already in use"
**Cause:** Another application is using port 8080.

**Fix:** Change the port in `application.properties`:
```properties
server.port=8081
```

Or kill the process using port 8080:
```bash
lsof -ti:8080 | xargs kill -9
```

### Error: "mvn: command not found"
**Cause:** Maven is not installed or not in PATH.

**Fix:** Install Maven:
```bash
sudo apt install maven
```

---

## Q&A

**Q: Why Java instead of Python for the API?**

**A:** Python is great for data science, but Java/Spring Boot is better for production APIs:
- **Performance:** Java is 10-50x faster for HTTP request handling
- **Type safety:** Catches errors at compile time
- **Scalability:** Handles thousands of concurrent requests efficiently
- **Ecosystem:** Spring Boot has built-in monitoring, security, caching

Many companies use Python for ML/analytics and Java for serving results.

**Q: What is Maven?**

**A:** Maven is a build tool for Java projects. It:
1. Downloads dependencies (like `pip install`)
2. Compiles your `.java` files to `.class` bytecode
3. Runs tests
4. Packages your app as a `.jar` file

Think of it as `pip` + `setuptools` + `pytest` combined.

**Q: What is a JAR file?**

**A:** A **JAR** (Java ARchive) is a ZIP file containing:
- Compiled `.class` files
- Resources (properties, JSON, images)
- Dependencies (libraries)

Spring Boot creates a "fat JAR" (or "uber JAR") that includes everything needed to run your app. You can deploy it with:
```bash
java -jar inflation-api-0.0.1-SNAPSHOT.jar
```

**Q: Why does Java have so many folders (src/main/java/com/cbn/inflationapi)?**

**A:** Java uses a strict package structure that mirrors the directory structure:
- Package name: `com.cbn.inflationapi`
- Directory path: `src/main/java/com/cbn/inflationapi/`

This prevents naming conflicts (e.g., two companies can have a `User` class in different packages).

**Q: Can I use Spring Boot 2 instead of Spring Boot 3?**

**A:** Yes, but Spring Boot 3 is the current major version (released Nov 2022). It requires Java 17 but has better performance and security. Stick with Spring Boot 3 unless you have a specific reason to use version 2.

**Q: What is @SpringBootApplication doing?**

**A:** It combines three annotations:
1. **@Configuration:** Declares this class as a configuration class
2. **@EnableAutoConfiguration:** Tells Spring Boot to guess what you need (e.g., "I see you have `spring-boot-starter-web`, so I'll start a Tomcat server")
3. **@ComponentScan:** Scans for `@Controller`, `@Service`, `@Repository` classes

It's a shortcut so you don't have to write all three annotations.

**Q: Why do I need a database if I already have JSON files?**

**A:** Databases are better for production:
- **Query flexibility:** `SELECT * FROM forecasts WHERE date > '2025-01-01'` vs. loading entire JSON file
- **Concurrency:** Multiple users can query simultaneously
- **Indexing:** Queries are fast even with millions of rows
- **Transactions:** Ensures data consistency

In Day 32-33, we'll load your Python results into PostgreSQL and query them via the API.

---

## What You Built Today

1. **Project skeleton:**
   - Maven project with proper directory structure
   - `pom.xml` with Spring Boot 3.2.2 and dependencies

2. **Main application class:**
   - Entry point with `@SpringBootApplication`
   - Starts embedded Tomcat server

3. **Configuration:**
   - `application.properties` with server port, logging, and DB placeholder

4. **Test endpoint:**
   - `/health` endpoint that returns JSON

**You now have a working Spring Boot application that responds to HTTP requests!**

---

## Next Steps

**Day 32:** You'll:
- Set up PostgreSQL database
- Create JPA entities (Java classes that map to database tables)
- Create a repository to query the database
- Build a REST endpoint to fetch forecast data

**Day 33:** You'll:
- Implement CRUD endpoints (Create, Read, Update, Delete)
- Add request validation
- Handle errors gracefully

By Day 35, you'll have a fully functional API that serves your VAR forecasts to a React dashboard.

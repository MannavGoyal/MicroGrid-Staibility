# Requirements Document

## Introduction

This document specifies requirements for restructuring and enhancing a microgrid stability prediction application. The system will transition from a monolithic Python/tkinter architecture to a modern Frontend-Backend architecture while significantly expanding its analytical capabilities. The enhanced system will support multiple prediction models, comprehensive stability analysis, and research-oriented comparative studies to quantify the impact of PV forecasting accuracy on microgrid stability.

## Glossary

- **Frontend**: React-based web application providing user interface for configuration, visualization, and analysis
- **Backend**: Python-based REST API server handling data processing, model training, and simulation
- **LSTM_Predictor**: Long Short-Term Memory neural network model for time-series PV power prediction
- **Classical_Predictor**: Traditional forecasting methods including Persistence, ARIMA, and SVR models
- **Advanced_Predictor**: Deep learning models including GRU, CNN-LSTM, and Transformer architectures
- **Microgrid_Simulator**: Physics-based simulation engine modeling microgrid component interactions
- **EMS_Controller**: Energy Management System controller optimizing battery dispatch based on forecasts
- **BESS**: Battery Energy Storage System
- **PV_Array**: Photovoltaic solar panel array
- **Inverter**: Power electronic device converting DC to AC power
- **Data_Pipeline**: Data preprocessing and feature engineering module
- **Stability_Analyzer**: Module calculating frequency, voltage, and power quality metrics
- **Comparative_Engine**: Module executing and comparing multiple forecasting approaches
- **Configuration_Manager**: Module handling user-defined simulation parameters
- **Results_Exporter**: Module formatting and exporting analysis results
- **Parser**: Module parsing configuration files and input data formats
- **Pretty_Printer**: Module formatting data structures into human-readable output
- **API_Gateway**: REST API endpoint handler routing requests to backend services

## Requirements

### Requirement 1: Architecture Restructuring

**User Story:** As a developer, I want the application separated into Frontend and Backend components, so that the system is maintainable, scalable, and follows modern architectural patterns.

#### Acceptance Criteria

1. THE Frontend SHALL be implemented as a React application in a dedicated `/frontend` directory
2. THE Backend SHALL be implemented as a Python REST API in a dedicated `/backend` directory
3. THE Frontend SHALL communicate with THE Backend exclusively through REST API endpoints
4. THE Backend SHALL expose API endpoints for model training, prediction, simulation, and results retrieval
5. THE Frontend SHALL NOT contain any machine learning or simulation logic
6. THE Backend SHALL NOT contain any UI rendering logic
7. THE Repository SHALL contain separate dependency management files for Frontend (package.json) and Backend (requirements.txt)

### Requirement 2: Configuration File Parsing

**User Story:** As a user, I want to define simulation parameters in configuration files, so that I can reproduce experiments and share configurations.

#### Acceptance Criteria

1. WHEN a valid JSON configuration file is provided, THE Parser SHALL parse it into a Configuration object
2. WHEN an invalid configuration file is provided, THE Parser SHALL return a descriptive error message indicating the validation failure
3. THE Configuration object SHALL contain fields for forecast_horizon, microgrid_mode, battery_capacity, model_type, and training_parameters
4. THE Pretty_Printer SHALL format Configuration objects into valid JSON configuration files
5. FOR ALL valid Configuration objects, parsing then printing then parsing SHALL produce an equivalent Configuration object (round-trip property)

### Requirement 3: Multiple Prediction Model Support

**User Story:** As a researcher, I want to compare multiple forecasting methods, so that I can quantify which approach provides the best stability improvements.

#### Acceptance Criteria

1. THE Backend SHALL support Classical_Predictor models including Persistence, ARIMA, and SVR
2. THE Backend SHALL support LSTM_Predictor as the baseline deep learning model
3. THE Backend SHALL support Advanced_Predictor models including GRU, CNN-LSTM, and Transformer architectures
4. WHEN a user selects a model type, THE Backend SHALL instantiate the corresponding predictor class
5. THE Backend SHALL maintain a consistent prediction interface across all model types accepting input features and returning forecasted PV power
6. WHEN training any predictor, THE Backend SHALL return training metrics including loss curves and convergence status

### Requirement 4: Configurable Forecast Horizons

**User Story:** As a researcher, I want to configure forecast horizons, so that I can study how prediction timeframes affect stability outcomes.

#### Acceptance Criteria

1. THE Configuration_Manager SHALL support forecast horizons of 5 minutes, 15 minutes, and 1 hour
2. WHEN a forecast horizon is selected, THE Data_Pipeline SHALL prepare sequences with the corresponding time steps
3. THE Data_Pipeline SHALL adjust input sequence length based on the selected forecast horizon to maintain temporal context
4. WHEN generating predictions, THE Backend SHALL output forecasts matching the configured horizon
5. THE Results_Exporter SHALL include the forecast horizon in all exported analysis reports

### Requirement 5: Enhanced Input Features

**User Story:** As a data scientist, I want to use comprehensive weather parameters as inputs, so that prediction accuracy improves beyond simple historical PV data.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL accept solar irradiance, temperature, cloud cover, humidity, and wind speed as input features
2. WHEN preprocessing data, THE Data_Pipeline SHALL normalize all input features to comparable scales
3. THE Data_Pipeline SHALL handle missing values through forward-fill or interpolation methods
4. THE Data_Pipeline SHALL create lagged features for temporal context spanning at least the sequence length
5. WHEN feature engineering is complete, THE Data_Pipeline SHALL output a feature matrix with shape (samples, sequence_length, num_features)

### Requirement 6: Microgrid Architecture Configuration

**User Story:** As a microgrid operator, I want to configure microgrid topology and operating mode, so that simulations reflect my specific system architecture.

#### Acceptance Criteria

1. THE Configuration_Manager SHALL support microgrid modes: grid-connected, islanded, and hybrid AC/DC
2. THE Configuration_Manager SHALL allow specification of components: PV_Array, Inverter, BESS, diesel generator (optional), loads, and EMS_Controller
3. WHEN islanded mode is selected, THE Microgrid_Simulator SHALL disable grid power exchange
4. WHEN grid-connected mode is selected, THE Microgrid_Simulator SHALL allow bidirectional power flow with the grid
5. WHERE a diesel generator is configured, THE Microgrid_Simulator SHALL include generator dispatch in the energy balance
6. THE Configuration_Manager SHALL validate that component ratings are physically consistent (e.g., inverter capacity ≥ PV array peak power)

### Requirement 7: Advanced Microgrid Simulation

**User Story:** As a researcher, I want realistic microgrid physics simulation, so that stability metrics accurately reflect real-world behavior.

#### Acceptance Criteria

1. THE Microgrid_Simulator SHALL model PV_Array output based on irradiance, temperature, and panel efficiency curves
2. THE Microgrid_Simulator SHALL model BESS dynamics including charge/discharge efficiency, capacity limits, and power limits
3. THE Microgrid_Simulator SHALL model Inverter behavior including efficiency curves and reactive power capability
4. WHEN simulating frequency dynamics, THE Microgrid_Simulator SHALL calculate frequency deviation based on power imbalance and system inertia
5. WHEN simulating voltage dynamics, THE Microgrid_Simulator SHALL calculate voltage deviation based on reactive power balance
6. THE Microgrid_Simulator SHALL enforce physical constraints: 0 ≤ SOC ≤ battery_capacity, frequency within ±2 Hz of nominal, voltage within ±10% of nominal

### Requirement 8: EMS Controller Integration

**User Story:** As a control engineer, I want the EMS to use forecasts for proactive battery dispatch, so that I can demonstrate forecast-driven stability improvements.

#### Acceptance Criteria

1. WHEN a PV forecast is available, THE EMS_Controller SHALL optimize battery dispatch to minimize frequency deviation over the forecast horizon
2. THE EMS_Controller SHALL implement Model Predictive Control (MPC) or rule-based control strategies
3. WHEN optimizing dispatch, THE EMS_Controller SHALL respect battery SOC limits, power limits, and charge/discharge efficiency
4. THE EMS_Controller SHALL update dispatch decisions at each simulation time step based on the latest forecast
5. WHEN no forecast is available, THE EMS_Controller SHALL operate in reactive mode responding only to current power imbalance
6. THE Microgrid_Simulator SHALL execute EMS_Controller dispatch commands and update system state accordingly

### Requirement 9: Comprehensive Stability Metrics

**User Story:** As a researcher, I want detailed stability metrics, so that I can quantify the relationship between forecast accuracy and grid stability.

#### Acceptance Criteria

1. THE Stability_Analyzer SHALL calculate frequency deviation metrics: mean absolute deviation, standard deviation, and maximum deviation in Hz
2. THE Stability_Analyzer SHALL calculate voltage deviation metrics: mean absolute deviation, standard deviation, and maximum deviation in percentage
3. THE Stability_Analyzer SHALL calculate battery stress metrics: SOC fluctuation range, number of charge/discharge cycles, and depth of discharge
4. THE Stability_Analyzer SHALL calculate power quality metrics: total harmonic distortion (THD) proxy and power factor
5. THE Stability_Analyzer SHALL calculate energy imbalance metrics: total unmet load and total curtailed PV energy
6. THE Stability_Analyzer SHALL calculate control effort metrics: sum of absolute battery power changes and number of control actions
7. FOR ALL stability metrics, THE Stability_Analyzer SHALL compute values for both forecast-enabled and no-forecast scenarios

### Requirement 10: Prediction Performance Metrics

**User Story:** As a data scientist, I want standard prediction metrics, so that I can evaluate and compare model performance.

#### Acceptance Criteria

1. THE Backend SHALL calculate Mean Absolute Error (MAE) between predicted and actual PV power
2. THE Backend SHALL calculate Root Mean Squared Error (RMSE) between predicted and actual PV power
3. THE Backend SHALL calculate Mean Absolute Percentage Error (MAPE) between predicted and actual PV power
4. THE Backend SHALL calculate R-squared (R²) coefficient of determination for predictions
5. THE Backend SHALL calculate prediction metrics separately for different times of day: morning, midday, afternoon, evening
6. THE Backend SHALL include confidence intervals or prediction uncertainty estimates where applicable

### Requirement 11: Comparative Analysis Framework

**User Story:** As a researcher, I want automated comparative analysis, so that I can systematically evaluate multiple forecasting approaches against baselines.

#### Acceptance Criteria

1. THE Comparative_Engine SHALL execute simulations for no-forecast baseline, Classical_Predictor methods, and Advanced_Predictor methods
2. THE Comparative_Engine SHALL use identical input data, microgrid configuration, and simulation parameters across all comparison runs
3. WHEN all simulations complete, THE Comparative_Engine SHALL generate a comparison table with prediction metrics and stability metrics for each method
4. THE Comparative_Engine SHALL calculate percentage improvement in each stability metric relative to the no-forecast baseline
5. THE Comparative_Engine SHALL identify the best-performing method for each metric category
6. THE Comparative_Engine SHALL generate statistical significance tests comparing methods where applicable

### Requirement 12: Data Requirements and Resolution

**User Story:** As a user, I want to provide historical data at appropriate resolution, so that the system can train accurate models and run realistic simulations.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL accept historical PV output data at 1-minute or 5-minute resolution
2. THE Data_Pipeline SHALL accept weather parameter data (irradiance, temperature, cloud cover, humidity, wind speed) at matching temporal resolution
3. THE Data_Pipeline SHALL accept load profile data at matching temporal resolution
4. WHERE frequency and voltage measurements are available, THE Data_Pipeline SHALL accept and incorporate them for validation
5. WHEN data resolution is coarser than 5 minutes, THE Data_Pipeline SHALL issue a warning about reduced prediction accuracy
6. THE Data_Pipeline SHALL validate that all input time series have matching timestamps and no gaps exceeding 10% of the dataset

### Requirement 13: Frontend Visualization

**User Story:** As a user, I want interactive visualizations, so that I can explore results and understand system behavior.

#### Acceptance Criteria

1. THE Frontend SHALL display time-series plots comparing actual vs predicted PV power
2. THE Frontend SHALL display battery SOC trajectories for forecast-enabled and no-forecast scenarios
3. THE Frontend SHALL display frequency deviation time series for both scenarios
4. THE Frontend SHALL display voltage deviation time series for both scenarios
5. THE Frontend SHALL provide interactive controls for zooming, panning, and selecting time ranges
6. THE Frontend SHALL display a metrics dashboard showing prediction and stability metrics in tabular format
7. THE Frontend SHALL allow users to toggle between different model comparison results
8. THE Frontend SHALL update visualizations dynamically when users change configuration parameters

### Requirement 14: Results Export

**User Story:** As a researcher, I want to export results in standard formats, so that I can include them in publications and further analysis.

#### Acceptance Criteria

1. THE Results_Exporter SHALL export visualizations as high-resolution PNG or SVG files
2. THE Results_Exporter SHALL export time-series data as CSV files with timestamps
3. THE Results_Exporter SHALL export metrics tables as CSV or JSON files
4. THE Results_Exporter SHALL export complete simulation configuration as JSON for reproducibility
5. THE Results_Exporter SHALL generate a summary report in Markdown or PDF format including all key findings
6. WHEN exporting, THE Results_Exporter SHALL create a timestamped directory containing all exported artifacts

### Requirement 15: Model Persistence

**User Story:** As a user, I want to save and load trained models, so that I can reuse models without retraining.

#### Acceptance Criteria

1. WHEN a model training completes, THE Backend SHALL save the trained model weights to disk
2. THE Backend SHALL save model metadata including architecture, hyperparameters, and training metrics
3. THE Backend SHALL save the fitted scaler or normalizer used during training
4. WHEN loading a saved model, THE Backend SHALL restore model weights, metadata, and scaler
5. THE Backend SHALL validate that loaded model architecture matches the saved configuration
6. THE Backend SHALL provide API endpoints for listing available saved models and loading specific models by identifier

### Requirement 16: API Design

**User Story:** As a frontend developer, I want well-defined REST API endpoints, so that I can integrate the frontend with backend services.

#### Acceptance Criteria

1. THE API_Gateway SHALL expose POST /api/train endpoint accepting configuration and returning training job identifier
2. THE API_Gateway SHALL expose GET /api/train/{job_id}/status endpoint returning training progress and metrics
3. THE API_Gateway SHALL expose POST /api/predict endpoint accepting model identifier and input data, returning predictions
4. THE API_Gateway SHALL expose POST /api/simulate endpoint accepting predictions and configuration, returning simulation results
5. THE API_Gateway SHALL expose GET /api/models endpoint returning list of available trained models
6. THE API_Gateway SHALL expose POST /api/compare endpoint accepting comparison configuration, returning comparative analysis results
7. THE API_Gateway SHALL expose GET /api/export/{result_id} endpoint returning downloadable result artifacts
8. THE API_Gateway SHALL return appropriate HTTP status codes: 200 for success, 400 for invalid input, 404 for not found, 500 for server errors
9. THE API_Gateway SHALL include error messages in response bodies for all error status codes

### Requirement 17: Error Handling

**User Story:** As a user, I want clear error messages, so that I can understand and fix issues quickly.

#### Acceptance Criteria

1. WHEN invalid configuration is provided, THE Backend SHALL return an error message specifying which parameter is invalid and why
2. WHEN model training fails, THE Backend SHALL return an error message indicating the failure reason (e.g., insufficient data, convergence failure)
3. WHEN simulation encounters physical constraint violations, THE Microgrid_Simulator SHALL log the violation and continue with constrained values
4. WHEN API requests fail, THE Frontend SHALL display user-friendly error messages with suggested corrective actions
5. THE Backend SHALL log all errors with timestamps, stack traces, and context information for debugging
6. IF a critical error occurs during long-running operations, THE Backend SHALL clean up partial results and release resources

### Requirement 18: Performance Requirements

**User Story:** As a user, I want responsive performance, so that I can iterate quickly on experiments.

#### Acceptance Criteria

1. WHEN training an LSTM model with 30 days of 5-minute resolution data, THE Backend SHALL complete training within 5 minutes on standard hardware
2. WHEN generating predictions for a test set, THE Backend SHALL process at least 1000 time steps per second
3. WHEN running a microgrid simulation, THE Microgrid_Simulator SHALL process at least 10000 time steps per second
4. THE Frontend SHALL render visualizations with up to 10000 data points within 2 seconds
5. THE API_Gateway SHALL respond to status check requests within 100 milliseconds
6. WHEN multiple users access the system concurrently, THE Backend SHALL handle at least 10 concurrent requests without degradation

### Requirement 19: Data Validation

**User Story:** As a user, I want automatic data validation, so that I can catch data quality issues before training or simulation.

#### Acceptance Criteria

1. WHEN data is uploaded, THE Data_Pipeline SHALL check for missing values and report the percentage of missing data
2. THE Data_Pipeline SHALL check for outliers using statistical methods (e.g., values beyond 3 standard deviations) and report them
3. THE Data_Pipeline SHALL validate that PV power values are non-negative and within physically plausible ranges
4. THE Data_Pipeline SHALL validate that temperature values are within reasonable ranges (-50°C to 60°C)
5. THE Data_Pipeline SHALL validate that irradiance values are between 0 and 1.2 kW/m²
6. IF data validation fails critical checks, THE Data_Pipeline SHALL reject the data and return a validation report
7. IF data validation fails non-critical checks, THE Data_Pipeline SHALL issue warnings but allow processing to continue

### Requirement 20: Testing and Validation

**User Story:** As a developer, I want comprehensive tests, so that I can ensure system correctness and prevent regressions.

#### Acceptance Criteria

1. THE Backend SHALL include unit tests for all predictor classes verifying prediction output shapes and value ranges
2. THE Backend SHALL include unit tests for THE Microgrid_Simulator verifying energy conservation and constraint enforcement
3. THE Backend SHALL include unit tests for THE Data_Pipeline verifying normalization and sequence generation
4. THE Backend SHALL include integration tests verifying end-to-end workflows from data input to results export
5. THE Backend SHALL include property-based tests verifying that round-trip parsing produces equivalent configurations
6. THE Frontend SHALL include component tests for all React components verifying rendering and user interactions
7. THE project SHALL achieve at least 80% code coverage for backend modules


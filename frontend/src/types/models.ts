/**
 * Type definitions for microgrid application models
 */

export type ForecastHorizon = '5min' | '15min' | '1hour'
export type MicrogridMode = 'grid_connected' | 'islanded' | 'hybrid_ac_dc'
export type ModelType = 'persistence' | 'arima' | 'svr' | 'lstm' | 'gru' | 'cnn_lstm' | 'transformer'

export interface ModelConfig {
  model_type: ModelType
  hyperparameters: Record<string, any>
  sequence_length: number
}

export interface MicrogridConfig {
  mode: MicrogridMode
  pv_capacity_kw: number
  battery_capacity_kwh: number
  battery_power_kw: number
  inverter_capacity_kw: number
  initial_soc_kwh?: number
  has_diesel_generator: boolean
  diesel_capacity_kw?: number
}

export interface TrainingConfig {
  epochs: number
  batch_size: number
  learning_rate: number
  validation_split: number
  early_stopping_patience: number
}

export interface Configuration {
  experiment_name: string
  forecast_horizon: ForecastHorizon
  model_config: ModelConfig
  microgrid_config: MicrogridConfig
  training_config: TrainingConfig
  data_path: string
  output_dir: string
}

export interface TrainRequest {
  config: Configuration
  data?: any
}

export interface TrainResponse {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  message: string
}

export interface TrainingStatus {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  progress: number
  current_epoch?: number
  metrics?: Record<string, number>
  error?: string
}

export interface PredictRequest {
  model_id: string
  input_data: number[][]
}

export interface PredictResponse {
  predictions: number[]
  confidence_intervals?: [number, number][]
}

export interface SimulateRequest {
  predictions: number[]
  actual_pv: number[]
  load_profile: number[]
  microgrid_config: MicrogridConfig
}

export interface SimulationResponse {
  result_id: string
  timeseries: Record<string, number[]>
  metrics: Record<string, any>
}

export interface CompareRequest {
  model_ids: string[]
  test_data_path: string
  microgrid_config: MicrogridConfig
}

export interface ComparisonResponse {
  comparison_id: string
  results: Record<string, any>
  rankings: Record<string, string[]>
  improvements: Record<string, any>
}

export interface ModelMetadata {
  model_id: string
  model_type: ModelType
  created_at: string
  metrics: {
    mae: number
    rmse: number
    r2: number
  }
}

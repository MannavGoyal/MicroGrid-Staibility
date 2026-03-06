/**
 * API service layer for backend communication
 */
import axios, { AxiosInstance } from 'axios'
import type {
  TrainRequest,
  TrainResponse,
  TrainingStatus,
  PredictRequest,
  PredictResponse,
  SimulateRequest,
  SimulationResponse,
  CompareRequest,
  ComparisonResponse,
  ModelMetadata,
} from '../types/models'

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          // Server responded with error status
          console.error('API Error:', error.response.data)
          throw new Error(error.response.data.error?.message || 'API request failed')
        } else if (error.request) {
          // Request made but no response
          console.error('Network Error:', error.request)
          throw new Error('Network error - backend may be unavailable')
        } else {
          // Something else happened
          console.error('Error:', error.message)
          throw error
        }
      }
    )
  }

  // Health check
  async healthCheck() {
    const response = await this.client.get('/health')
    return response.data
  }

  // Training endpoints
  async startTraining(request: TrainRequest): Promise<TrainResponse> {
    const response = await this.client.post('/train', request)
    return response.data
  }

  async getTrainingStatus(jobId: string): Promise<TrainingStatus> {
    const response = await this.client.get(`/train/${jobId}/status`)
    return response.data
  }

  async cancelTraining(jobId: string): Promise<void> {
    await this.client.delete(`/train/${jobId}`)
  }

  // Model management endpoints
  async listModels(modelType?: string, sortBy?: string): Promise<{ models: ModelMetadata[] }> {
    const params = new URLSearchParams()
    if (modelType) params.append('model_type', modelType)
    if (sortBy) params.append('sort_by', sortBy)
    
    const response = await this.client.get('/models', { params })
    return response.data
  }

  async getModelDetails(modelId: string): Promise<ModelMetadata> {
    const response = await this.client.get(`/models/${modelId}`)
    return response.data
  }

  async deleteModel(modelId: string): Promise<void> {
    await this.client.delete(`/models/${modelId}`)
  }

  async loadModel(modelId: string): Promise<void> {
    await this.client.post(`/models/${modelId}/load`)
  }

  // Prediction endpoints
  async predict(request: PredictRequest): Promise<PredictResponse> {
    const response = await this.client.post('/predict', request)
    return response.data
  }

  // Simulation endpoints
  async simulate(request: SimulateRequest): Promise<SimulationResponse> {
    const response = await this.client.post('/simulate', request)
    return response.data
  }

  // Comparison endpoints
  async startComparison(request: CompareRequest): Promise<ComparisonResponse> {
    const response = await this.client.post('/compare', request)
    return response.data
  }

  async getComparisonStatus(comparisonId: string): Promise<any> {
    const response = await this.client.get(`/compare/${comparisonId}/status`)
    return response.data
  }

  async getComparisonResults(comparisonId: string): Promise<any> {
    const response = await this.client.get(`/compare/${comparisonId}/results`)
    return response.data
  }

  // Export endpoints
  async exportResults(resultId: string, format: string, include: string): Promise<Blob> {
    const response = await this.client.get(`/export/${resultId}`, {
      params: { format, include },
      responseType: 'blob',
    })
    return response.data
  }

  // Data management endpoints
  async uploadData(file: File): Promise<{ data_id: string; validation_report: any }> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await this.client.post('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  async validateData(dataPath: string): Promise<any> {
    const response = await this.client.post('/data/validate', { data_path: dataPath })
    return response.data
  }

  async getDataMetadata(dataId: string): Promise<any> {
    const response = await this.client.get(`/data/${dataId}`)
    return response.data
  }
}

export const apiService = new ApiService()
export default apiService

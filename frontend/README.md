# Microgrid Frontend

React + TypeScript frontend application for microgrid stability prediction and analysis.

## Structure

```
frontend/
├── src/
│   ├── components/        # React components
│   │   ├── ConfigurationPanel.tsx
│   │   ├── VisualizationDashboard.tsx
│   │   ├── MetricsTable.tsx
│   │   ├── TimeSeriesChart.tsx
│   │   └── ModelComparison.tsx
│   ├── services/          # API service layer
│   │   └── api.ts
│   ├── types/             # TypeScript type definitions
│   │   └── models.ts
│   ├── test/              # Test setup
│   │   └── setup.ts
│   ├── App.tsx            # Main application component
│   ├── main.tsx           # Entry point
│   └── index.css          # Global styles
├── package.json           # Dependencies
├── tsconfig.json          # TypeScript config
├── vite.config.ts         # Vite config
└── tailwind.config.js     # Tailwind CSS config
```

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

## Development

Start development server:
```bash
npm run dev
```

Application will run on `http://localhost:3000`

The dev server includes a proxy to the backend API at `http://localhost:5000`

## Building

Build for production:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Testing

Run tests:
```bash
npm run test
```

Run tests in watch mode:
```bash
npm run test:watch
```

## Code Quality

Lint code:
```bash
npm run lint
```

Type check:
```bash
npm run type-check
```

## Components

Components will be implemented in Task 18:
- **ConfigurationPanel**: Form for experiment configuration
- **TimeSeriesChart**: Interactive time-series plots
- **MetricsTable**: Prediction and stability metrics display
- **VisualizationDashboard**: Battery SOC, frequency, voltage plots
- **ModelComparison**: Multi-model comparison results

## API Integration

All backend communication goes through `src/services/api.ts`:
- Centralized error handling
- Type-safe requests/responses
- Automatic request/response transformation

Example usage:
```typescript
import { apiService } from './services/api'

// Check backend health
const status = await apiService.healthCheck()

// Start training
const response = await apiService.startTraining(trainRequest)

// Get predictions
const predictions = await apiService.predict(predictRequest)
```

## Styling

Uses Tailwind CSS for styling. Customize theme in `tailwind.config.js`.

## State Management

Uses Zustand for lightweight state management (to be implemented in Task 18).

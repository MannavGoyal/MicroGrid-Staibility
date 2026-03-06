import { useState } from 'react'

function App() {
  const [status, setStatus] = useState<string>('Not connected')

  const checkBackendHealth = async () => {
    console.log('Checking backend health...')
    setStatus('Checking...')
    try {
      const response = await fetch('/health')
      console.log('Response:', response)
      const data = await response.json()
      console.log('Data:', data)
      setStatus(`Connected: ${data.service} - ${data.status}`)
    } catch (error) {
      console.error('Error:', error)
      setStatus('Backend connection failed')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-gray-900">
            Microgrid Stability Enhancement
          </h1>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <div className="px-4 py-6 sm:px-0">
            <div className="border-4 border-dashed border-gray-200 rounded-lg p-8">
              <h2 className="text-xl font-semibold mb-4">System Status</h2>
              <p className="mb-4">Backend Status: {status}</p>
              <button
                onClick={checkBackendHealth}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              >
                Check Backend Connection
              </button>
              <div className="mt-8">
                <p className="text-gray-600">
                  Frontend and Backend architecture successfully set up.
                  Components will be added in subsequent tasks.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

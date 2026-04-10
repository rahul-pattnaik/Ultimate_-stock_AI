import React, { useState, useEffect } from 'react';

export default function StockAIDashboard() {
  const [stocks, setStocks] = useState([
    { symbol: 'AAPL', name: 'Apple', price: 189.45, change: 2.34, sector: 'Technology', pe: 28.5 },
    { symbol: 'MSFT', name: 'Microsoft', price: 371.82, change: 1.87, sector: 'Technology', pe: 32.1 },
    { symbol: 'GOOGL', name: 'Google', price: 139.67, change: -0.45, sector: 'Technology', pe: 24.3 },
    { symbol: 'AMZN', name: 'Amazon', price: 176.94, change: 3.21, sector: 'Consumer', pe: 61.2 },
    { symbol: 'TSLA', name: 'Tesla', price: 242.84, change: -2.15, sector: 'Automotive', pe: 95.4 },
  ]);

  const [selectedStock, setSelectedStock] = useState(stocks[0]);
  const [activeTab, setActiveTab] = useState('overview');
  const [metrics, setMetrics] = useState({
    rsi: 65.4,
    macd: 'BULLISH',
    volatility: 18.5,
    valuation: 'UNDERVALUED',
    momentum: 12.3
  });

  return (
    <div className="min-h-screen" style={{
      background: 'linear-gradient(135deg, #0f172a 0%, #1a2a4a 50%, #0d1b2a 100%)',
      fontFamily: "'Outfit', 'SF Pro Display', system-ui, sans-serif",
      color: '#e8eef2'
    }}>
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-8 right-5 w-72 h-72 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse" style={{animationDelay: '2s'}}></div>
      </div>

      <div className="relative z-10">
        {/* Header */}
        <header className="backdrop-blur-md border-b border-blue-400/20 sticky top-0">
          <div className="max-w-7xl mx-auto px-8 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-200 via-cyan-300 to-blue-400 bg-clip-text text-transparent">
                  StockAI Pro
                </h1>
                <p className="text-sm text-blue-300/70 mt-1">Advanced AI-Powered Stock Analysis Engine</p>
              </div>
              <div className="flex items-center gap-4">
                <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-400 to-cyan-500 flex items-center justify-center text-white font-bold">
                  ⚡
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-8 py-8">
          <div className="grid grid-cols-12 gap-6">
            {/* Left Sidebar - Stock List */}
            <div className="col-span-3">
              <div className="backdrop-blur-xl bg-blue-900/20 border border-blue-400/30 rounded-2xl p-6 sticky top-24">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <span className="w-2 h-2 bg-cyan-400 rounded-full"></span>
                  Watchlist
                </h2>
                
                <div className="space-y-2">
                  {stocks.map((stock) => (
                    <button
                      key={stock.symbol}
                      onClick={() => setSelectedStock(stock)}
                      className={`w-full p-4 rounded-xl transition-all duration-300 text-left ${
                        selectedStock.symbol === stock.symbol
                          ? 'bg-gradient-to-r from-blue-500 to-cyan-500 shadow-lg shadow-cyan-500/50'
                          : 'bg-blue-800/30 hover:bg-blue-700/40'
                      }`}
                    >
                      <div className="font-semibold text-sm">{stock.symbol}</div>
                      <div className="text-xs text-blue-200/70">{stock.sector}</div>
                      <div className="flex justify-between items-end mt-2">
                        <span className="text-lg font-bold">${stock.price}</span>
                        <span className={`text-xs font-semibold ${stock.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {stock.change > 0 ? '+' : ''}{stock.change}%
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Main Content Area */}
            <div className="col-span-9 space-y-6">
              {/* Stock Header */}
              <div className="backdrop-blur-xl bg-gradient-to-br from-blue-900/40 to-cyan-900/20 border border-blue-400/30 rounded-2xl p-8">
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h1 className="text-5xl font-bold tracking-tight mb-2">{selectedStock.symbol}</h1>
                    <p className="text-blue-300/80">{selectedStock.name}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-5xl font-bold text-cyan-400">${selectedStock.price}</div>
                    <div className={`text-2xl font-semibold mt-2 flex items-center justify-end gap-2 ${selectedStock.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {selectedStock.change > 0 ? '↑' : '↓'} {Math.abs(selectedStock.change)}%
                    </div>
                  </div>
                </div>

                {/* Key Metrics Grid */}
                <div className="grid grid-cols-4 gap-4">
                  <div className="bg-blue-800/40 rounded-xl p-4">
                    <p className="text-xs text-blue-300/60 uppercase tracking-wider mb-2">P/E Ratio</p>
                    <p className="text-2xl font-bold">{selectedStock.pe}</p>
                  </div>
                  <div className="bg-blue-800/40 rounded-xl p-4">
                    <p className="text-xs text-blue-300/60 uppercase tracking-wider mb-2">52W High</p>
                    <p className="text-2xl font-bold text-emerald-400">$245.32</p>
                  </div>
                  <div className="bg-blue-800/40 rounded-xl p-4">
                    <p className="text-xs text-blue-300/60 uppercase tracking-wider mb-2">Volume</p>
                    <p className="text-2xl font-bold">52.3M</p>
                  </div>
                  <div className="bg-blue-800/40 rounded-xl p-4">
                    <p className="text-xs text-blue-300/60 uppercase tracking-wider mb-2">Market Cap</p>
                    <p className="text-2xl font-bold">$2.8T</p>
                  </div>
                </div>
              </div>

              {/* Tabs */}
              <div className="flex gap-2 backdrop-blur-xl bg-blue-900/20 border border-blue-400/30 rounded-xl p-2">
                {['overview', 'technical', 'valuation', 'peers'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`flex-1 py-3 rounded-lg font-semibold text-sm transition-all duration-300 ${
                      activeTab === tab
                        ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg shadow-cyan-500/50'
                        : 'text-blue-300 hover:text-blue-100'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>

              {/* Tab Content */}
              {activeTab === 'overview' && (
                <div className="backdrop-blur-xl bg-blue-900/20 border border-blue-400/30 rounded-2xl p-8">
                  <h3 className="text-2xl font-bold mb-6">Technical Analysis</h3>
                  
                  <div className="grid grid-cols-2 gap-6">
                    <div className="bg-blue-800/30 rounded-xl p-6 border border-blue-400/20">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-blue-300/80 font-semibold">RSI (14)</span>
                        <span className="text-3xl font-bold text-cyan-400">{metrics.rsi}</span>
                      </div>
                      <div className="w-full bg-blue-900/50 rounded-full h-2">
                        <div className="bg-gradient-to-r from-cyan-500 to-blue-500 h-2 rounded-full" style={{width: `${metrics.rsi}%`}}></div>
                      </div>
                      <p className="text-xs text-blue-300/60 mt-2">Approaching overbought (>70)</p>
                    </div>

                    <div className="bg-blue-800/30 rounded-xl p-6 border border-blue-400/20">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-blue-300/80 font-semibold">MACD Signal</span>
                        <span className="text-2xl font-bold text-emerald-400">{metrics.macd}</span>
                      </div>
                      <p className="text-sm text-emerald-300">Bullish crossover detected</p>
                      <p className="text-xs text-blue-300/60 mt-2">Positive momentum building</p>
                    </div>

                    <div className="bg-blue-800/30 rounded-xl p-6 border border-blue-400/20">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-blue-300/80 font-semibold">Volatility (30d)</span>
                        <span className="text-3xl font-bold text-orange-400">{metrics.volatility}%</span>
                      </div>
                      <p className="text-xs text-blue-300/60">Moderate volatility range</p>
                    </div>

                    <div className="bg-blue-800/30 rounded-xl p-6 border border-blue-400/20">
                      <div className="flex justify-between items-center mb-4">
                        <span className="text-blue-300/80 font-semibold">AI Valuation</span>
                        <span className="text-2xl font-bold text-cyan-400">{metrics.valuation}</span>
                      </div>
                      <p className="text-sm text-cyan-300">+12.4% upside potential</p>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'technical' && (
                <div className="backdrop-blur-xl bg-blue-900/20 border border-blue-400/30 rounded-2xl p-8">
                  <h3 className="text-2xl font-bold mb-4">Advanced Indicators</h3>
                  <p className="text-blue-300/80">40+ Technical Indicators Ready for Analysis</p>
                  <div className="mt-6 p-6 bg-blue-800/40 rounded-xl border border-blue-400/20">
                    <div className="space-y-2">
                      {['SMA 20/50/200', 'EMA 12/26', 'Bollinger Bands', 'Stochastic Oscillator', 'ATR', 'ADX', 'OBV', 'Williams %R'].map((ind) => (
                        <div key={ind} className="flex items-center justify-between">
                          <span className="text-blue-300">{ind}</span>
                          <span className="text-xs bg-cyan-500/30 text-cyan-400 px-2 py-1 rounded">Active</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'valuation' && (
                <div className="backdrop-blur-xl bg-blue-900/20 border border-blue-400/30 rounded-2xl p-8">
                  <h3 className="text-2xl font-bold mb-6">Valuation Models</h3>
                  <div className="grid grid-cols-2 gap-6">
                    <div className="bg-gradient-to-br from-blue-800/50 to-cyan-800/30 rounded-xl p-6 border border-cyan-400/30">
                      <h4 className="font-semibold text-cyan-300 mb-4">DCF Analysis</h4>
                      <p className="text-3xl font-bold text-emerald-400">$195.42</p>
                      <p className="text-xs text-blue-300/60 mt-2">Intrinsic Value</p>
                    </div>
                    <div className="bg-gradient-to-br from-blue-800/50 to-cyan-800/30 rounded-xl p-6 border border-cyan-400/30">
                      <h4 className="font-semibold text-cyan-300 mb-4">Comparable Multiples</h4>
                      <p className="text-3xl font-bold text-emerald-400">$192.18</p>
                      <p className="text-xs text-blue-300/60 mt-2">Peer-based Valuation</p>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'peers' && (
                <div className="backdrop-blur-xl bg-blue-900/20 border border-blue-400/30 rounded-2xl p-8">
                  <h3 className="text-2xl font-bold mb-6">Peer Comparison</h3>
                  <div className="space-y-3">
                    {stocks.slice(0, 4).map((stock) => (
                      <div key={stock.symbol} className="flex items-center justify-between p-4 bg-blue-800/30 rounded-xl border border-blue-400/20">
                        <div>
                          <p className="font-semibold">{stock.symbol}</p>
                          <p className="text-xs text-blue-300/60">{stock.sector}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-bold">${stock.price}</p>
                          <p className={`text-xs ${stock.change >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {stock.change > 0 ? '+' : ''}{stock.change}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-blue-400/20 backdrop-blur-md mt-12">
          <div className="max-w-7xl mx-auto px-8 py-6 text-center text-blue-300/60 text-sm">
            <p>StockAI Pro © 2024 | Real-time AI-powered analysis | Not financial advice</p>
          </div>
        </footer>
      </div>
    </div>
  );
}

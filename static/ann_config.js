/**
 * Enhanced ANN Configuration UI Component v3.0
 * Comprehensive neural network architecture customization with
 * live visualization, training progress simulation, and advanced settings
 */

class ANNConfigurator {
    constructor() {
        this.layers = [
            { neurons: 128, dropout: 0.3, activation: 'relu', batch_norm: true },
            { neurons: 64, dropout: 0.2, activation: 'relu', batch_norm: true },
            { neurons: 32, dropout: 0.1, activation: 'relu', batch_norm: true }
        ];
        this.maxLayers = 10;
        this.maxNeurons = 1024;
        this.minNeurons = 4;
        this.colors = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308'];
    }

    render(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="space-y-4">
                <!-- Header with controls -->
                <div class="flex justify-between items-center flex-wrap gap-3">
                    <div>
                        <h3 class="font-bold text-lg" style="background: linear-gradient(135deg, #667eea, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            <i class="fas fa-network-wired mr-2"></i> Network Architecture
                        </h3>
                        <p class="text-xs text-gray-500">Customize hidden layers, activations & regularization</p>
                    </div>
                    <div class="flex gap-2">
                        <button onclick="annConfig.addLayer()" class="btn-sm px-3 py-1.5 rounded text-white text-xs font-medium"
                                style="background: linear-gradient(135deg, #10b981, #059669); transition: all 0.2s;"
                                onmouseover="this.style.boxShadow='0 4px 12px rgba(16,185,129,0.4)'"
                                onmouseout="this.style.boxShadow='none'">
                            <i class="fas fa-plus mr-1"></i> Add Layer
                        </button>
                        <button onclick="annConfig.removeLayer()" class="btn-sm px-3 py-1.5 rounded text-white text-xs font-medium"
                                style="background: linear-gradient(135deg, #ef4444, #dc2626); transition: all 0.2s;"
                                onmouseover="this.style.boxShadow='0 4px 12px rgba(239,68,68,0.4)'"
                                onmouseout="this.style.boxShadow='none'"
                                ${this.layers.length <= 1 ? 'disabled style="opacity:0.5;cursor:not-allowed;"' : ''}>
                            <i class="fas fa-minus mr-1"></i> Remove
                        </button>
                        <button onclick="annConfig.resetArchitecture()" class="btn-sm px-3 py-1.5 rounded text-white text-xs font-medium"
                                style="background: linear-gradient(135deg, #f59e0b, #d97706); transition: all 0.2s;"
                                onmouseover="this.style.boxShadow='0 4px 12px rgba(245,158,11,0.4)'"
                                onmouseout="this.style.boxShadow='none'">
                            <i class="fas fa-redo mr-1"></i> Reset
                        </button>
                    </div>
                </div>

                <!-- Architecture Canvas -->
                <div class="rounded-xl overflow-hidden border" style="background: #0f172a; border-color: rgba(99,102,241,0.3);">
                    <div class="p-3 flex justify-between items-center" style="background: rgba(99,102,241,0.08); border-bottom: 1px solid rgba(99,102,241,0.15);">
                        <span class="text-xs font-medium text-indigo-300">
                            <i class="fas fa-eye mr-1"></i> Network Visualization
                        </span>
                        <div class="flex gap-3 text-xs">
                            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full" style="background:#10b981;"></span> Input</span>
                            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full" style="background:#667eea;"></span> Hidden</span>
                            <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full" style="background:#f59e0b;"></span> Output</span>
                        </div>
                    </div>
                    <div class="p-2 flex justify-center">
                        <canvas id="annCanvas" width="750" height="320" style="max-width:100%; height:auto;"></canvas>
                    </div>
                </div>

                <!-- Layer Cards -->
                <div id="layersContainer" class="space-y-2.5">
                    ${this.layers.map((layer, idx) => this.renderLayer(layer, idx)).join('')}
                </div>

                <!-- Architecture Stats Summary -->
                <div class="rounded-xl border p-4" style="background: rgba(99,102,241,0.05); border-color: rgba(99,102,241,0.2);">
                    <div class="grid grid-cols-4 gap-4 text-center">
                        <div>
                            <div class="text-2xl font-bold text-indigo-400">${this.layers.length}</div>
                            <div class="text-xs text-gray-500">Hidden Layers</div>
                        </div>
                        <div>
                            <div class="text-2xl font-bold text-purple-400">${this.layers.reduce((s, l) => s + l.neurons, 0)}</div>
                            <div class="text-xs text-gray-500">Total Neurons</div>
                        </div>
                        <div>
                            <div class="text-2xl font-bold text-green-400">${this.layers.some(l => l.batch_norm) ? 'Yes' : 'No'}</div>
                            <div class="text-xs text-gray-500">Batch Norm</div>
                        </div>
                        <div>
                            <div class="text-xl font-bold text-amber-400 truncate" title="${this.layers.map(l => l.activation).join(', ')}">
                                ${this.layers[0].activation}
                                ${this.layers.every(l => l.activation === this.layers[0].activation) ? '' : '+...'}
                            </div>
                            <div class="text-xs text-gray-500">Activation</div>
                        </div>
                    </div>
                </div>

                <!-- Architecture Flow Summary -->
                <div class="rounded-xl border p-4" style="background: #0f172a; border-color: rgba(99,102,241,0.25);">
                    <h4 class="font-semibold mb-2 text-indigo-300 text-sm flex items-center gap-2">
                        <i class="fas fa-code-branch"></i> Architecture Flow
                    </h4>
                    <div id="architectureSummary" class="font-mono text-sm leading-relaxed">
                        ${this.getSummary()}
                    </div>
                </div>

                <!-- Training Progress Indicator (hidden by default) -->
                <div id="trainingProgress" class="hidden rounded-xl border p-4" style="background: rgba(16,185,129,0.05); border-color: rgba(16,185,129,0.25);">
                    <h4 class="font-semibold mb-3 text-green-400 flex items-center gap-2">
                        <i class="fas fa-circle-notch fa-spin"></i> Training Progress
                    </h4>
                    <div class="space-y-3">
                        <div>
                            <div class="flex justify-between text-xs mb-1">
                                <span style="color: var(--text-secondary);">Epoch Progress</span>
                                <span style="color: var(--text-secondary);"><span id="currentEpoch">0</span> / <span id="totalEpochs">0</span></span>
                            </div>
                            <div class="progress-bar">
                                <div id="progressFill" class="progress-fill" style="width: 0%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-4 text-xs">
                            <div class="bg-black/20 rounded p-2">
                                <span class="text-gray-500 block">Current Loss</span>
                                <span id="currentLoss" class="text-green-400 font-mono text-sm">-</span>
                            </div>
                            <div class="bg-black/20 rounded p-2">
                                <span class="text-gray-500 block">Validation Loss</span>
                                <span id="valLoss" class="text-amber-400 font-mono text-sm">-</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Preset Architectures -->
                <div class="flex flex-wrap gap-2 pt-1">
                    ${[
                        { name: 'Shallow', layers: [{n:64, d:0.2}], icon: 'fa-layer-group' },
                        { name: 'Medium', layers: [{n:128, d:0.3}, {n:64, d:0.2}], icon: 'fa-layer-group' },
                        { name: 'Deep', layers: [{n:256, d:0.4}, {n:128, d:0.3}, {n:64, d:0.2}], icon: 'fa-layer-group' },
                        { name: 'Wide', layers: [{n:512, d:0.5}, {n:256, d:0.4}, {n:128, d:0.3}, {n:64, d:0.2}], icon: 'fa-layer-group' },
                    ].map(p => `
                        <span class="model-chip text-xs" onclick="annConfig.applyPreset('${p.name}')">
                            <i class="fas ${p.icon}"></i> ${p.name}
                        </span>
                    `).join('')}
                    <span class="text-xs text-gray-500 italic ml-1">— presets</span>
                </div>
            </div>
        `;

        this.drawVisualization();
    }

    renderLayer(layer, idx) {
        const actOptions = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'gelu'];
        return `
            <div class="rounded-xl border overflow-hidden transition-all duration-200 hover:border-indigo-400/50"
                 style="background: rgba(99,102,241,0.04); border-color: rgba(99,102,241,0.15);">
                <!-- Header -->
                <div class="px-4 py-2.5 flex justify-between items-center" 
                     style="background: linear-gradient(90deg, rgba(99,102,241,0.08), transparent); border-bottom: 1px solid rgba(99,102,241,0.1);">
                    <div class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full" style="background: ${this.colors[idx % this.colors.length]};"></span>
                        <span class="font-semibold text-sm">Hidden Layer ${idx + 1}</span>
                        <span class="badge-pill text-xs" style="background: rgba(99,102,241,0.2); color: #a5b4fc;">
                            ${layer.neurons} neurons
                        </span>
                    </div>
                    <div class="flex items-center gap-2">
                        <label class="text-xs text-gray-500 flex items-center gap-1.5 cursor-pointer">
                            <input type="checkbox" ${layer.batch_norm ? 'checked' : ''}
                                   onchange="annConfig.updateLayer(${idx}, 'batch_norm', this.checked)"
                                   class="form-checkbox rounded" style="accent-color: #667eea;">
                            <span>BN</span>
                        </label>
                        <i class="fas fa-grip-vertical text-gray-600 text-xs cursor-move"></i>
                    </div>
                </div>

                <!-- Controls -->
                <div class="p-4">
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <!-- Neurons -->
                        <div>
                            <label class="text-xs block mb-1.5 text-gray-500 font-medium">Neurons</label>
                            <div class="flex items-center gap-2">
                                <input type="range" min="${this.minNeurons}" max="${this.maxNeurons}" step="8" 
                                       value="${layer.neurons}" 
                                       oninput="annConfig.updateLayer(${idx}, 'neurons', parseInt(this.value)); document.getElementById('neuron_val_${idx}').textContent = this.value;"
                                       class="flex-1" style="accent-color: #667eea;">
                                <span id="neuron_val_${idx}" class="text-xs font-mono text-indigo-400 min-w-[3rem] text-right">${layer.neurons}</span>
                            </div>
                        </div>
                        <!-- Dropout -->
                        <div>
                            <label class="text-xs block mb-1.5 text-gray-500 font-medium">Dropout</label>
                            <div class="flex items-center gap-2">
                                <input type="range" min="0" max="0.6" step="0.05" 
                                       value="${layer.dropout}" 
                                       oninput="annConfig.updateLayer(${idx}, 'dropout', parseFloat(this.value)); document.getElementById('dropout_val_${idx}').textContent = this.value;"
                                       class="flex-1" style="accent-color: #a855f7;">
                                <span id="dropout_val_${idx}" class="text-xs font-mono text-purple-400 min-w-[3rem] text-right">${layer.dropout}</span>
                            </div>
                        </div>
                        <!-- Activation -->
                        <div>
                            <label class="text-xs block mb-1.5 text-gray-500 font-medium">Activation</label>
                            <select onchange="annConfig.updateLayer(${idx}, 'activation', this.value)" 
                                    class="w-full text-xs rounded-lg px-2 py-1.5"
                                    style="background: rgba(255,255,255,0.08); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.12);">
                                ${actOptions.map(a => `<option value="${a}" ${layer.activation === a ? 'selected' : ''}>${a.toUpperCase().replace('_', ' ')}</option>`).join('')}
                            </select>
                        </div>
                        <!-- Layer Info -->
                        <div class="flex items-end">
                            <div class="text-xs text-gray-500 bg-black/20 rounded-lg p-2 w-full text-center">
                                <span class="block text-indigo-300 font-semibold">${layer.neurons * (idx + 1) * 4 + layer.neurons * layer.neurons * (idx > 0 ? 1 : 0)}</span>
                                <span class="text-gray-600">params (est.)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    updateLayer(idx, field, value) {
        this.layers[idx][field] = value;
        const summary = document.getElementById('architectureSummary');
        if (summary) summary.innerHTML = this.getSummary();
        this.drawVisualization();
    }

    addLayer() {
        if (this.layers.length >= this.maxLayers) {
            this.showToast(`Maximum ${this.maxLayers} layers allowed`, 'warning');
            return;
        }
        const lastLayer = this.layers[this.layers.length - 1];
        const newNeurons = Math.max(this.minNeurons, Math.floor(lastLayer.neurons / 1.5));
        this.layers.push({
            neurons: newNeurons,
            dropout: Math.min(0.5, lastLayer.dropout + 0.05),
            activation: lastLayer.activation,
            batch_norm: true
        });
        this.render('annConfigContainer');
        this.showToast(`Added hidden layer ${this.layers.length} (${newNeurons} neurons)`, 'success');
    }

    removeLayer() {
        if (this.layers.length <= 1) {
            this.showToast('Minimum 1 hidden layer required', 'warning');
            return;
        }
        this.layers.pop();
        this.render('annConfigContainer');
        this.showToast('Layer removed', 'info');
    }

    resetArchitecture() {
        this.layers = [
            { neurons: 128, dropout: 0.3, activation: 'relu', batch_norm: true },
            { neurons: 64, dropout: 0.2, activation: 'relu', batch_norm: true },
            { neurons: 32, dropout: 0.1, activation: 'relu', batch_norm: true }
        ];
        this.render('annConfigContainer');
        this.showToast('Architecture reset to default', 'info');
    }

    applyPreset(name) {
        const presets = {
            'Shallow': [{n:64, d:0.2}],
            'Medium': [{n:128, d:0.3}, {n:64, d:0.2}],
            'Deep': [{n:256, d:0.4}, {n:128, d:0.3}, {n:64, d:0.2}],
            'Wide': [{n:512, d:0.5}, {n:256, d:0.4}, {n:128, d:0.3}, {n:64, d:0.2}]
        };
        const preset = presets[name];
        if (preset) {
            this.layers = preset.map(l => ({
                neurons: l.n,
                dropout: l.d,
                activation: 'relu',
                batch_norm: true
            }));
            this.render('annConfigContainer');
            this.showToast(`Applied "${name}" architecture (${this.layers.length} layers)`, 'success');
        }
    }

    getSummary() {
        const total = this.layers.reduce((sum, l) => sum + l.neurons, 0);
        const params = this.layers.reduce((sum, l, i) => {
            const prev = i === 0 ? 0 : this.layers[i-1].neurons;
            return sum + (prev * l.neurons + l.neurons); // weights + biases
        }, 0);
        const flow = this.layers.map(l => 
            `<span class="text-indigo-300">${l.neurons}</span><span class="text-gray-600 text-xs">(${l.activation}${l.batch_norm ? '+BN' : ''}${l.dropout > 0 ? ',drop='+l.dropout : ''})</span>`
        ).join(' <span class="text-gray-700">→</span> ');
        return `
            <div class="space-y-1.5">
                <div class="flex items-center gap-2 text-sm">
                    <span class="text-green-400 font-medium">Input</span>
                    <span class="text-gray-600">→</span>
                    ${flow}
                    <span class="text-gray-600">→</span>
                    <span class="text-amber-400 font-medium">Output</span>
                </div>
                <div class="text-xs text-gray-500 flex gap-4 mt-2">
                    <span>🧠 <strong class="text-indigo-300">${total}</strong> hidden neurons</span>
                    <span>📊 <strong class="text-purple-300">${this.layers.length}</strong> layers</span>
                    <span>⚙️ <strong class="text-green-300">${params.toLocaleString()}</strong> estimated params</span>
                    <span>📉 dropout: <strong class="text-amber-300">${this.layers.map(l => (l.dropout*100).toFixed(0)+'%').join(', ')}</strong></span>
                </div>
            </div>
        `;
    }

    drawVisualization() {
        const canvas = document.getElementById('annCanvas');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);

        const totalLayers = this.layers.length;
        const layerSpacing = width / (totalLayers + 2);
        const inputX = layerSpacing;
        const outputX = width - layerSpacing;

        // Draw connections first (behind nodes)
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.08)';
        ctx.lineWidth = 1;

        // Input to first hidden
        const firstX = inputX;
        const secondX = inputX + layerSpacing;
        const inputNodes = 5;
        const firstNodes = Math.min(8, Math.max(3, Math.ceil(this.layers[0].neurons / 24)));
        
        for (let n1 = 0; n1 < inputNodes; n1++) {
            for (let n2 = 0; n2 < firstNodes; n2++) {
                const y1 = height/2 - (inputNodes-1)*20 + n1*40;
                const y2 = height/2 - (firstNodes-1)*20 + n2*40;
                ctx.beginPath();
                ctx.moveTo(firstX + 18, y1);
                ctx.lineTo(secondX - 18, y2);
                ctx.stroke();
            }
        }

        // Between hidden layers
        for (let i = 0; i < totalLayers - 1; i++) {
            const x1 = inputX + (i + 1) * layerSpacing;
            const x2 = inputX + (i + 2) * layerSpacing;
            const nodes1 = Math.min(8, Math.max(3, Math.ceil(this.layers[i].neurons / 24)));
            const nodes2 = Math.min(8, Math.max(3, Math.ceil(this.layers[i+1].neurons / 24)));
            
            for (let n1 = 0; n1 < nodes1; n1++) {
                for (let n2 = 0; n2 < nodes2; n2++) {
                    const y1 = height/2 - (nodes1-1)*20 + n1*40;
                    const y2 = height/2 - (nodes2-1)*20 + n2*40;
                    ctx.beginPath();
                    ctx.moveTo(x1 + 18, y1);
                    ctx.lineTo(x2 - 18, y2);
                    ctx.stroke();
                }
            }
        }

        // Last hidden to output
        const lastX = inputX + totalLayers * layerSpacing;
        const outputNodes = 3;
        const lastNodes = Math.min(8, Math.max(3, Math.ceil(this.layers[totalLayers-1].neurons / 24)));
        for (let n1 = 0; n1 < lastNodes; n1++) {
            for (let n2 = 0; n2 < outputNodes; n2++) {
                const y1 = height/2 - (lastNodes-1)*20 + n1*40;
                const y2 = height/2 - (outputNodes-1)*20 + n2*40;
                ctx.beginPath();
                ctx.moveTo(lastX + 18, y1);
                ctx.lineTo(outputX - 18, y2);
                ctx.stroke();
            }
        }

        // Draw layers
        this.drawLayerColumn(ctx, inputX, height/2, inputNodes, 'Input', '#10b981', 14);
        
        this.layers.forEach((layer, idx) => {
            const x = inputX + (idx + 1) * layerSpacing;
            const numNodes = Math.min(8, Math.max(3, Math.ceil(layer.neurons / 24)));
            const color = this.colors[idx % this.colors.length];
            this.drawLayerColumn(ctx, x, height/2, numNodes, `H${idx+1}\n${layer.neurons}`, color, 14);
        });

        this.drawLayerColumn(ctx, outputX, height/2, outputNodes, 'Output', '#f59e0b', 14);
    }

    drawLayerColumn(ctx, x, centerY, numNodes, label, color, nodeRadius) {
        const nodeSpacing = 40;
        const startY = centerY - (numNodes - 1) * nodeSpacing / 2;

        // Draw glow effect
        for (let i = 0; i < numNodes; i++) {
            const y = startY + i * nodeSpacing;
            
            // Glow
            const gradient = ctx.createRadialGradient(x, y, nodeRadius, x, y, nodeRadius * 2.5);
            gradient.addColorStop(0, color + '30');
            gradient.addColorStop(1, 'transparent');
            ctx.beginPath();
            ctx.arc(x, y, nodeRadius * 2.5, 0, 2 * Math.PI);
            ctx.fillStyle = gradient;
            ctx.fill();

            // Node
            ctx.beginPath();
            ctx.arc(x, y, nodeRadius, 0, 2 * Math.PI);
            const nodeGradient = ctx.createRadialGradient(x-3, y-3, 2, x, y, nodeRadius);
            nodeGradient.addColorStop(0, '#ffffff');
            nodeGradient.addColorStop(0.3, color);
            nodeGradient.addColorStop(1, color + 'aa');
            ctx.fillStyle = nodeGradient;
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }

        // Label
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const lines = label.split('\n');
        lines.forEach((line, idx) => {
            ctx.fillText(line, x, startY - nodeRadius - 15 - idx * 14);
        });
    }

    getConfig() {
        return {
            layers: this.layers,
            epochs: parseInt(document.getElementById('annEpochs')?.value || 100),
            batch_size: parseInt(document.getElementById('annBatchSize')?.value || 32),
            learning_rate: parseFloat(document.getElementById('annLearningRate')?.value || 0.001),
            optimizer: document.getElementById('annOptimizer')?.value || 'adamw'
        };
    }

    showProgress(current, total, loss) {
        const progressContainer = document.getElementById('trainingProgress');
        if (progressContainer) {
            progressContainer.classList.remove('hidden');
            const progressFill = document.getElementById('progressFill');
            const currentEpoch = document.getElementById('currentEpoch');
            const totalEpochs = document.getElementById('totalEpochs');
            const currentLoss = document.getElementById('currentLoss');
            
            if (progressFill) progressFill.style.width = `${(current / total * 100)}%`;
            if (currentEpoch) currentEpoch.textContent = current;
            if (totalEpochs) totalEpochs.textContent = total;
            if (currentLoss) currentLoss.textContent = loss ? loss.toFixed(6) : '-';
        }
    }

    hideProgress() {
        const progressContainer = document.getElementById('trainingProgress');
        if (progressContainer) {
            progressContainer.classList.add('hidden');
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        const colors = {
            success: 'bg-green-500', info: 'bg-blue-500', warning: 'bg-amber-500', error: 'bg-red-500'
        };
        const icons = {
            success: 'fa-check-circle', info: 'fa-info-circle', warning: 'fa-exclamation-triangle', error: 'fa-times-circle'
        };
        toast.className = `fixed top-4 right-4 ${colors[type] || 'bg-blue-500'} text-white px-4 py-3 rounded-lg shadow-2xl z-50 flex items-center gap-2 text-sm font-medium`;
        toast.style.animation = 'slideInRight 0.3s ease-out';
        toast.innerHTML = `<i class="fas ${icons[type]}"></i> ${message}`;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => toast.remove(), 300);
        }, 2500);
    }
}

// Global instance
const annConfig = new ANNConfigurator();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        const container = document.getElementById('annConfigContainer');
        if (container) annConfig.render('annConfigContainer');
    });
} else {
    const container = document.getElementById('annConfigContainer');
    if (container) annConfig.render('annConfigContainer');
}
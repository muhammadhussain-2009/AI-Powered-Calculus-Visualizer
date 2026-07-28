/**
 * Calculus Visualizer - Frontend Logic
 * Configured with full Desmos Graphing Calculator API (v1.8) capabilities,
 * real-time expression sync, auto-viewport bounds, and math engine integration.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('visualization-form');
    const promptInput = document.getElementById('prompt-input');
    const submitBtn = document.getElementById('submit-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const explanationCard = document.getElementById('explanation-card');
    const visTitle = document.getElementById('vis-title');
    const visExplanation = document.getElementById('vis-explanation');
    const visTime = document.getElementById('vis-time');
    const visCount = document.getElementById('vis-count');
    const expressionList = document.getElementById('expression-list');
    const clearBtn = document.getElementById('clear-btn');
    const resetViewBtn = document.getElementById('reset-view-btn');
    const exportPngBtn = document.getElementById('export-png-btn');
    const keypadToggleBtn = document.getElementById('keypad-toggle-btn');
    const angleModeBtn = document.getElementById('angle-mode-btn');
    const apiStatusText = document.getElementById('api-status-text');

    let calculator = null;
    let jwtToken = null;
    let isKeypadVisible = true;
    let isDegreeMode = false;

    // 1. Initialize Desmos Calculator with Full Desmos API Features
    function initDesmosCalculator() {
        const elt = document.getElementById('desmos-calculator');
        if (!elt) {
            console.error('Element #desmos-calculator not found.');
            return;
        }

        if (typeof Desmos === 'undefined') {
            console.error('Desmos API script not loaded.');
            elt.innerHTML = '<div style="color:#f43f5e; padding:40px; text-align:center; font-weight:600;">Failed to load Desmos Graphing API. Please check your internet connection.</div>';
            return;
        }

        try {
            // Configure Desmos Graphing Calculator to full capabilities
            calculator = Desmos.GraphingCalculator(elt, {
                keypad: true,
                expressions: true,
                settingsMenu: true,
                zoomButtons: true,
                expressionsCollapsed: false,
                autosize: true,
                border: false,
                lockViewport: false,
                degreeMode: false, // Default Radians mode for Calculus
                showGrid: true,
                showXAxis: true,
                showYAxis: true,
                xAxisNumbers: true,
                yAxisNumbers: true,
                trace: true,
                pasteGraphpaperBounds: true
            });

            console.log('Desmos Graphing Calculator initialized with full feature set.');

            // Listen to real-time expression changes inside Desmos Calculator
            calculator.observe('expressions', () => {
                syncExpressionsFromCalculator();
            });

            // Handle window resize events for Desmos calculator
            window.addEventListener('resize', () => {
                if (calculator) calculator.resize();
            });

            // Set initial default visualization
            renderDefaultGraph();

        } catch (err) {
            console.error('Failed to initialize Desmos calculator:', err);
        }
    }

    // 2. Establish Session
    async function initSession() {
        try {
            const res = await fetch('/api/auth/session');
            if (res.ok) {
                const data = await res.json();
                jwtToken = data.access_token;
                if (apiStatusText) apiStatusText.textContent = 'Graphing Engine Active';
            }
        } catch (err) {
            console.warn('Could not establish session token:', err);
            if (apiStatusText) apiStatusText.textContent = 'Standalone Mode';
        }
    }

    // 3. Render Default Starting Graph
    function renderDefaultGraph() {
        if (!calculator) return;
        calculator.setBlank();
        calculator.setExpression({ id: 'exp_f', latex: 'y = \\sin(x)', color: '#2d70b3', lineWidth: 3 });
        calculator.setExpression({ id: 'exp_tangent', latex: 'y = x', color: '#38bdf8', lineStyle: Desmos.Styles.DASHED, lineWidth: 2 });
        calculator.setMathBounds({ left: -10, right: 10, bottom: -5, top: 5 });
    }

    // 4. Handle Visualization Request to Backend
    async function handleVisualize(promptText) {
        if (!promptText || !promptText.trim()) return;

        setLoading(true);

        try {
            const headers = { 'Content-Type': 'application/json' };
            if (jwtToken) {
                headers['Authorization'] = `Bearer ${jwtToken}`;
            }

            const response = await fetch('/api/visualize', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ prompt: promptText.trim() })
            });

            const result = await response.json();

            if (!result.success) {
                showError(result.error || 'Visualization request failed.');
                setLoading(false);
                return;
            }

            renderVisualization(result.data, result.processing_time_ms);

        } catch (err) {
            console.error('API Error:', err);
            showError('Network error connecting to Calculus Visualizer.');
        } finally {
            setLoading(false);
        }
    }

    // 5. Inject Expressions into Desmos Calculator and Auto-Fit Bounds
    function renderVisualization(data, processingTimeMs) {
        if (!calculator || !data) return;

        calculator.setBlank();

        const expressions = data.expressions || [];

        expressions.forEach((exp) => {
            if (!exp.latex) return;

            const desmosPayload = {
                id: exp.id || 'exp_' + Math.random().toString(36).substr(2, 9),
                latex: exp.latex,
                color: exp.color || '#2d70b3',
                hidden: exp.hidden || false,
                secret: exp.secret || false
            };

            if (exp.lineStyle === 'DASHED') {
                desmosPayload.lineStyle = Desmos.Styles.DASHED;
            } else if (exp.lineStyle === 'DOTTED') {
                desmosPayload.lineStyle = Desmos.Styles.DOTTED;
            }

            if (exp.lineWidth) {
                desmosPayload.lineWidth = exp.lineWidth;
            }

            if (exp.label) {
                desmosPayload.label = exp.label;
                desmosPayload.showLabel = exp.showLabel !== undefined ? exp.showLabel : true;
            }

            if (exp.sliderBounds) {
                desmosPayload.sliderBounds = {
                    min: exp.sliderBounds.min,
                    max: exp.sliderBounds.max,
                    step: exp.sliderBounds.step
                };
            }

            calculator.setExpression(desmosPayload);
        });

        // Auto-Adjust Math Bounds if line/curve is outside standard bounds (e.g. y = 3x + 34)
        autoFitBoundsForExpressions(expressions);

        // Update Explanation Card
        if (visTitle) visTitle.textContent = data.title || 'Calculus Graph';
        if (visExplanation) visExplanation.textContent = data.concept_explanation || '';
        if (visTime) visTime.textContent = `⏱️ ${processingTimeMs ? processingTimeMs.toFixed(0) : '0'} ms`;
        if (visCount) visCount.textContent = `📊 ${expressions.length} Curve${expressions.length !== 1 ? 's' : ''}`;
        if (explanationCard) explanationCard.classList.remove('hidden');

        updateSidebarExpressionDrawer(expressions);
    }

    // Auto-Fit Desmos Viewport Bounds for Large Off-Screen Linear/Polynomial Curves
    function autoFitBoundsForExpressions(expressions) {
        if (!calculator || !expressions || expressions.length === 0) return;

        for (const exp of expressions) {
            const latex = exp.latex || '';
            // Check for linear equation like y = 3x + 34 or 2x + 3y = 34
            const linearMatch = latex.match(/^y\s*=\s*(-?\d*(?:\.\d+)?)\s*x\s*([\+\-]\s*\d+(?:\.\d+)?)$/i);
            if (linearMatch) {
                const m = parseFloat(linearMatch[1]) || 1;
                const bStr = linearMatch[2].replace(/\s+/g, '');
                const b = parseFloat(bStr) || 0;

                if (Math.abs(b) > 15) {
                    const yCenter = b;
                    const yRange = Math.max(30, Math.abs(b) * 1.5);
                    calculator.setMathBounds({
                        left: -20,
                        right: 20,
                        bottom: yCenter - yRange,
                        top: yCenter + yRange
                    });
                    return;
                }
            }

            // Check for constant line y = C
            const constMatch = latex.match(/^y\s*=\s*(-?\d+(?:\.\d+)?)$/i);
            if (constMatch) {
                const c = parseFloat(constMatch[1]);
                if (Math.abs(c) > 10) {
                    calculator.setMathBounds({
                        left: -15,
                        right: 15,
                        bottom: c - 20,
                        top: c + 20
                    });
                    return;
                }
            }
        }

        // Default calculus view bounds
        calculator.setMathBounds({ left: -10, right: 10, bottom: -10, top: 10 });
    }

    // 6. Sync Expressions directly from Desmos Calculator into Sidebar Drawer
    function syncExpressionsFromCalculator() {
        if (!calculator) return;
        const currentExps = calculator.getExpressions();
        const activeExps = currentExps.filter(e => e.latex && e.latex.trim().length > 0);

        if (visCount) {
            visCount.textContent = `📊 ${activeExps.length} Curve${activeExps.length !== 1 ? 's' : ''}`;
        }

        updateSidebarExpressionDrawer(activeExps);
    }

    // 7. Update Expression Drawer UI
    function updateSidebarExpressionDrawer(expressions) {
        if (!expressionList) return;
        expressionList.innerHTML = '';

        const validExps = expressions.filter(e => e.latex && e.latex.trim().length > 0);

        if (validExps.length === 0) {
            expressionList.innerHTML = '<li class="empty-state">No expressions active. Type in Desmos calculator or submit a prompt above.</li>';
            return;
        }

        validExps.forEach((exp) => {
            const li = document.createElement('li');
            li.className = 'expression-item';
            li.innerHTML = `
                <span class="color-badge" style="background-color: ${exp.color || '#2d70b3'}"></span>
                <span class="exp-latex">${escapeHtml(exp.latex)}</span>
            `;
            expressionList.appendChild(li);
        });
    }

    // Helpers
    function setLoading(isLoading) {
        if (submitBtn) submitBtn.disabled = isLoading;
        document.querySelectorAll('.chip').forEach((chip) => {
            chip.disabled = isLoading;
        });
        if (btnSpinner) {
            if (isLoading) btnSpinner.classList.remove('hidden');
            else btnSpinner.classList.add('hidden');
        }
    }

    function showError(msg) {
        if (visTitle) visTitle.textContent = 'Notice';
        if (visExplanation) visExplanation.textContent = msg;
        if (visTime) visTime.textContent = '⚠️ Info';
        if (visCount) visCount.textContent = '📊 0 Curves';
        if (explanationCard) explanationCard.classList.remove('hidden');
    }

    function escapeHtml(str) {
        return (str || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // Event Listeners
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            if (promptInput && promptInput.value.trim()) {
                handleVisualize(promptInput.value);
            }
        });
    }

    // Preset Chips
    document.querySelectorAll('.chip').forEach((chip) => {
        chip.addEventListener('click', () => {
            if (chip.disabled) return;
            const promptText = chip.getAttribute('data-prompt');
            if (promptInput && promptText) {
                promptInput.value = promptText;
                handleVisualize(promptText);
            }
        });
    });

    // Clear Button
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (calculator) calculator.setBlank();
            if (expressionList) expressionList.innerHTML = '<li class="empty-state">Graph cleared.</li>';
            if (explanationCard) explanationCard.classList.add('hidden');
        });
    }

    // Reset View Button
    if (resetViewBtn) {
        resetViewBtn.addEventListener('click', () => {
            if (calculator) {
                calculator.setMathBounds({ left: -10, right: 10, bottom: -10, top: 10 });
            }
        });
    }

    // Keypad Toggle Button
    if (keypadToggleBtn) {
        keypadToggleBtn.addEventListener('click', () => {
            if (!calculator) return;
            isKeypadVisible = !isKeypadVisible;
            calculator.updateSettings({ keypad: isKeypadVisible });
            keypadToggleBtn.style.opacity = isKeypadVisible ? '1' : '0.6';
        });
    }

    // Angle Mode Toggle Button (Radians / Degrees)
    if (angleModeBtn) {
        angleModeBtn.addEventListener('click', () => {
            if (!calculator) return;
            isDegreeMode = !isDegreeMode;
            calculator.updateSettings({ degreeMode: isDegreeMode });
            angleModeBtn.textContent = isDegreeMode ? 'Degrees' : 'Radians';
        });
    }

    // Export Image Button (High-Res 1080p Screenshot)
    if (exportPngBtn) {
        exportPngBtn.addEventListener('click', () => {
            if (!calculator) return;
            calculator.asyncScreenshot({ width: 1920, height: 1080, targetPixelRatio: 2 }, (dataUrl) => {
                const a = document.createElement('a');
                a.href = dataUrl;
                a.download = 'calculus_graph_visualization.png';
                a.click();
            });
        });
    }

    // Initialize
    initDesmosCalculator();
    initSession();
});

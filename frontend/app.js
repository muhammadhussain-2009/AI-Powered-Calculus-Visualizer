/**
 * Samjh AI - Frontend Logic
 * Configured with full Desmos Graphing Calculator API (v1.8) capabilities,
 * LangGraph SSE real-time streaming, deliberate thinking delay UX, and auto-viewport bounds.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('visualization-form');
    const promptInput = document.getElementById('prompt-input');
    const clearPromptBtn = document.getElementById('clear-prompt-btn');
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
    const DELIBERATE_DELAY_MS = 600; // 600ms step-by-step thinking delay

    const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    /**
     * Fetch wrapper with AbortController 10-second timeout, up to 2 retries,
     * short retry delay, and clean error reporting to prevent app hanging.
     */
    async function fetchWithTimeoutAndRetry(url, options = {}, maxRetries = 2, timeoutMs = 10000, retryDelayMs = 1000) {
        let lastError = null;

        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

            try {
                const fetchOptions = {
                    ...options,
                    signal: controller.signal
                };

                const response = await fetch(url, fetchOptions);
                clearTimeout(timeoutId);

                if (response.ok) {
                    return response;
                }

                const statusText = `HTTP Error ${response.status}: ${response.statusText}`;
                lastError = new Error(statusText);
                console.warn(`[API Attempt ${attempt + 1}/${maxRetries + 1} Failed] ${url}: ${statusText}`);
            } catch (err) {
                clearTimeout(timeoutId);
                if (err.name === 'AbortError') {
                    lastError = new Error(`Request timed out after ${timeoutMs / 1000}s`);
                    console.warn(`[API Attempt ${attempt + 1}/${maxRetries + 1} Timeout] ${url} timed out after ${timeoutMs}ms.`);
                } else {
                    lastError = err;
                    console.warn(`[API Attempt ${attempt + 1}/${maxRetries + 1} Error] ${url}: ${err.message}`);
                }
            }

            if (attempt < maxRetries) {
                await sleep(retryDelayMs);
            }
        }

        throw lastError || new Error(`API call to ${url} failed after ${maxRetries + 1} attempts.`);
    }

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

    // 2. Establish Session with Retry & 10s Timeout
    async function initSession() {
        try {
            const res = await fetchWithTimeoutAndRetry('/api/auth/session', { method: 'GET' }, 2, 10000, 800);
            if (res.ok) {
                const data = await res.json();
                jwtToken = data.access_token;
                if (apiStatusText) apiStatusText.textContent = 'Graphing Engine Active';
            }
        } catch (err) {
            console.warn('Could not establish session token after retries:', err);
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

    // 4. Handle Streaming Visualization Request (LangGraph SSE + 10s Timeout & Retry UX)
    async function handleVisualize(promptText) {
        if (!promptText || !promptText.trim()) return;

        setLoading(true);

        if (visTitle) visTitle.textContent = 'AI Math Reasoning...';
        if (visExplanation) visExplanation.textContent = 'Node 1: Deconstructing calculus request into mathematical components...';
        if (visTime) visTime.textContent = '⏱️ Streaming...';
        if (visCount) visCount.textContent = '📊 0 Curves';
        if (explanationCard) explanationCard.classList.remove('hidden');
        if (expressionList) expressionList.innerHTML = '<li class="empty-state">Building graph step-by-step...</li>';

        const startTime = Date.now();
        const injectedExpressions = [];

        try {
            const headers = { 'Content-Type': 'application/json' };
            if (jwtToken) {
                headers['Authorization'] = `Bearer ${jwtToken}`;
            }

            const response = await fetchWithTimeoutAndRetry('/api/visualize/stream', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ prompt: promptText.trim() })
            }, 2, 10000, 1000);

            if (!response.body) {
                throw new Error('Response body stream unreadable.');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop() || ''; // Keep partial line in buffer

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('data: ')) {
                        const jsonStr = trimmed.slice(6).trim();
                        if (!jsonStr) continue;

                        try {
                            const eventData = JSON.parse(jsonStr);

                            if (eventData.type === 'error') {
                                showError(eventData.error || 'Visualization request failed.');
                                setLoading(false);
                                return;
                            }

                            if (eventData.type === 'analysis') {
                                if (visExplanation) {
                                    renderMathText(visExplanation, `Node 1 Analysis: ${eventData.intent || 'Analyzed intent.'}`);
                                }
                            } else if (eventData.type === 'metadata') {
                                if (visTitle) renderMathText(visTitle, eventData.title || 'Calculus Graph');
                                if (visExplanation) renderMathText(visExplanation, eventData.concept_explanation || '');
                            } else if (eventData.type === 'expression') {
                                const exp = eventData.expression;
                                if (exp && exp.latex) {
                                    // Inject single expression into Desmos
                                    injectSingleExpression(exp);
                                    injectedExpressions.push(exp);

                                    if (visCount) {
                                        visCount.textContent = `📊 ${injectedExpressions.length} Curve${injectedExpressions.length !== 1 ? 's' : ''}`;
                                    }
                                    updateSidebarExpressionDrawer(injectedExpressions);
                                    autoFitBoundsForExpressions(injectedExpressions);

                                    // Deliberate Thinking UX Delay (600ms between commands)
                                    await sleep(DELIBERATE_DELAY_MS);
                                }
                            } else if (eventData.type === 'complete') {
                                const elapsedTime = Date.now() - startTime;
                                if (visTime) visTime.textContent = `⏱️ ${elapsedTime} ms`;
                                // Trigger Visual Feedback Loop (capture Base64 screenshot & send to /api/verify-graph)
                                sendVisualFeedback(promptText);
                            }
                        } catch (parseErr) {
                            console.warn('Failed to parse SSE JSON:', jsonStr, parseErr);
                        }
                    }
                }
            }

        } catch (err) {
            console.warn('Streaming error after retries, falling back to standard API endpoint:', err);
            await handleVisualizeFallback(promptText);
        } finally {
            setLoading(false);
        }
    }

    // 5. Fallback API Handler for Non-Streaming Response with 10s Timeout & Retry
    async function handleVisualizeFallback(promptText) {
        try {
            const headers = { 'Content-Type': 'application/json' };
            if (jwtToken) {
                headers['Authorization'] = `Bearer ${jwtToken}`;
            }

            const response = await fetchWithTimeoutAndRetry('/api/visualize', {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ prompt: promptText.trim() })
            }, 2, 10000, 1000);

            const result = await response.json();

            if (!result.success) {
                showError(result.error || 'Visualization request failed.');
                return;
            }

            await renderVisualizationSequential(result.data, result.processing_time_ms, promptText);

        } catch (err) {
            console.error('API Error after retries:', err);
            showError('Request timed out or failed to connect to Samjh AI backend. Please check your network connection and try again.');
        }
    }

    // 6. Sequential Render with Deliberate Thinking Delay (400ms - 800ms)
    async function renderVisualizationSequential(data, processingTimeMs, promptText) {
        if (!calculator || !data) return;

        const expressions = data.expressions || [];
        const injectedExpressions = [];

        if (visTitle) renderMathText(visTitle, data.title || 'Calculus Graph');
        if (visExplanation) renderMathText(visExplanation, data.concept_explanation || '');
        if (visTime) visTime.textContent = `⏱️ ${processingTimeMs ? processingTimeMs.toFixed(0) : '0'} ms`;
        if (explanationCard) explanationCard.classList.remove('hidden');

        for (let i = 0; i < expressions.length; i++) {
            const exp = expressions[i];
            if (!exp.latex) continue;

            injectSingleExpression(exp);
            injectedExpressions.push(exp);

            if (visCount) visCount.textContent = `📊 ${injectedExpressions.length} Curve${injectedExpressions.length !== 1 ? 's' : ''}`;
            updateSidebarExpressionDrawer(injectedExpressions);
            autoFitBoundsForExpressions(injectedExpressions);

            if (i < expressions.length - 1) {
                await sleep(DELIBERATE_DELAY_MS); // Deliberate Thinking UX Delay
            }
        }

        // Trigger Visual Feedback Loop (capture Base64 screenshot & send to /api/verify-graph)
        sendVisualFeedback(promptText || data.title);
    }

    // Visual Feedback Loop: Automatically captures Base64 PNG screenshot of rendered Desmos graph & sends to /api/verify-graph with 10s Timeout & Retry
    function sendVisualFeedback(promptText) {
        if (!calculator) return;

        try {
            calculator.asyncScreenshot({ width: 800, height: 600, targetPixelRatio: 1 }, async (dataUrl) => {
                if (!dataUrl) return;

                const headers = { 'Content-Type': 'application/json' };
                if (jwtToken) {
                    headers['Authorization'] = `Bearer ${jwtToken}`;
                }

                try {
                    const response = await fetchWithTimeoutAndRetry('/api/verify-graph', {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify({
                            image: dataUrl,
                            prompt: promptText ? promptText.trim() : '',
                            expressions_count: calculator.getExpressions().length
                        })
                    }, 2, 10000, 1000);
                    const resJson = await response.json();
                    console.log('Visual Feedback Loop (/api/verify-graph):', resJson);
                } catch (err) {
                    console.warn('Failed to send visual feedback screenshot after retries:', err);
                }
            });
        } catch (err) {
            console.warn('Could not capture Desmos screenshot for visual feedback:', err);
        }
    }

    // Inject Single Expression into Desmos Calculator
    function injectSingleExpression(exp) {
        if (!calculator || !exp || !exp.latex) return;

        const uniqueId = exp.id ? `${exp.id}_${Math.random().toString(36).substr(2, 6)}` : 'exp_' + Math.random().toString(36).substr(2, 9);
        const desmosPayload = {
            id: uniqueId,
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
    }

    // Auto-Fit Desmos Viewport Bounds for Large Off-Screen Linear/Polynomial Curves
    function autoFitBoundsForExpressions(expressions) {
        if (!calculator || !expressions || expressions.length === 0) return;

        for (const exp of expressions) {
            const latex = exp.latex || '';
            const linearMatch = latex.match(/^y\s*=\s*(-?\d*(?:\.\d+)?)\s*x\s*([\+\-]\s*\d+(?:\.\d+)?)$/i);
            if (linearMatch) {
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

    // 7. Sync Expressions directly from Desmos Calculator into Sidebar Drawer
    function syncExpressionsFromCalculator() {
        if (!calculator) return;
        const currentExps = calculator.getExpressions();
        const activeExps = currentExps.filter(e => e.latex && e.latex.trim().length > 0);

        if (visCount) {
            visCount.textContent = `📊 ${activeExps.length} Curve${activeExps.length !== 1 ? 's' : ''}`;
        }

        updateSidebarExpressionDrawer(activeExps);
    }

    // 8. Update Expression Drawer UI with KaTeX Math Formatting
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

            const badge = document.createElement('span');
            badge.className = 'color-badge';
            badge.style.backgroundColor = exp.color || '#2d70b3';

            const latexSpan = document.createElement('span');
            latexSpan.className = 'exp-latex';
            const formattedLatex = asciiToLatex(exp.latex);
            latexSpan.innerHTML = window.katex && exp.latex ? katex.renderToString(formattedLatex, { displayMode: false, throwOnError: false }) : escapeHtml(exp.latex);

            li.appendChild(badge);
            li.appendChild(latexSpan);
            expressionList.appendChild(li);
        });
    }

    /**
     * Converts raw ASCII mathematical expressions to valid LaTeX syntax.
     * Handles exponents (x^3, x^(3/2)), fractions (dy/dx, a/b), functions (sin, cos, ln, sqrt),
     * and symbols (∫, d/dx).
     */
    function asciiToLatex(mathStr) {
        if (!mathStr) return '';
        let s = mathStr.trim();

        // 1. Convert derivative notation: dy/dx, d/dx, dz/dt
        s = s.replace(/\b(d[a-z]?)\/(d[a-z])\b/gi, '\\frac{$1}{$2}');

        // 2. Convert trig & math functions if not already LaTeX escaped
        s = s.replace(/(?<!\\)\b(sin|cos|tan|sec|csc|cot|sinh|cosh|tanh|ln|log|exp)\b/gi, '\\$1');
        s = s.replace(/(?<!\\)\bsqrt\(([^)]+)\)/gi, '\\sqrt{$1}');

        // 3. Convert exponents with parenthesized power: x^(3/2) -> x^{3/2}, e^(2x) -> e^{2x}
        s = s.replace(/([a-zA-Z0-9\)\}\_]+)\^\(([^)]+)\)/g, '$1^{$2}');

        // 4. Convert exponents without parens: x^3 -> x^{3}, x^2 -> x^{2}, x^-1 -> x^{-1}
        s = s.replace(/([a-zA-Z0-9\)\}\_]+)\^([a-zA-Z0-9\+\-]+)/g, '$1^{$2}');

        // 5. Convert simple fraction parens: (a)/(b) -> \frac{a}{b}
        s = s.replace(/\(([^)]+)\)\/\(([^)]+)\)/g, '\\frac{$1}{$2}');

        // 6. Convert multiplication symbol * to \cdot
        s = s.replace(/(\d+)\s*\*\s*([a-zA-Z])/g, '$1 \\cdot $2');

        // 7. Convert integral symbol ∫ to \int
        s = s.replace(/∫/g, '\\int ');

        return s;
    }

    /**
     * Intercepts UI text and renders mathematical equations, superscripts (x^3), fractions,
     * and formulas using KaTeX textbook math formatting.
     */
    function renderMathText(element, rawText) {
        if (!element) return;
        if (!rawText || !rawText.trim()) {
            element.textContent = '';
            return;
        }

        const text = rawText.trim();

        if (typeof katex === 'undefined') {
            element.textContent = text;
            return;
        }

        try {
            // Case 1: Text contains explicit LaTeX delimiters ($...$ or $$...$$)
            if (text.includes('$')) {
                element.innerHTML = text.replace(/\$\$([^$]+)\$\$/g, (m, math) => {
                    return katex.renderToString(asciiToLatex(math), { displayMode: true, throwOnError: false });
                }).replace(/\$([^$]+)\$/g, (m, math) => {
                    return katex.renderToString(asciiToLatex(math), { displayMode: false, throwOnError: false });
                });
                return;
            }

            // Case 2: Implicit ASCII / LaTeX math parsing without explicit $ delimiters
            // Matches mathematical equations, functions, derivatives, exponents, fractions, and intervals:
            // e.g., f(x) = x^3 - 3x, y = 3x + 34, dy/dx = x + y, x^(3/2), x^3, [-2, 2], f(x), \sin(x)
            const mathRegex = /(?:\\(?:frac|sin|cos|tan|sec|csc|cot|ln|log|sqrt|int|sum|lim|partial|infty|alpha|beta|theta|pi|cdot)[\{a-zA-Z0-9_\^\+\-\*\/\s\(\)]*|\b[fg]\([xyt]\)\s*=\s*[^,;\.\?\!\n]+|\b[a-zA-Z0-9_]+\s*=\s*[-+\d\.\s*a-zA-Z^\(\)\/]+|\b[a-zA-Z0-9_]+\^[a-zA-Z0-9\(\)\+\-\/]+|\bd[a-z]?\/d[a-z]\b|\b[fg]'\([xyt]\)\b|\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]|\b(?:sin|cos|tan|ln|log|sqrt)\([^\)]+\)|\b[fg]\([xyt]\))/gi;

            let lastIndex = 0;
            let resultHtml = '';
            let match;

            while ((match = mathRegex.exec(text)) !== null) {
                if (match.index > lastIndex) {
                    resultHtml += escapeHtml(text.slice(lastIndex, match.index));
                }

                const rawMath = match[0];
                const latex = asciiToLatex(rawMath);
                try {
                    const rendered = katex.renderToString(latex, { displayMode: false, throwOnError: false });
                    resultHtml += rendered;
                } catch (kErr) {
                    resultHtml += escapeHtml(rawMath);
                }

                lastIndex = mathRegex.lastIndex;
            }

            if (lastIndex < text.length) {
                resultHtml += escapeHtml(text.slice(lastIndex));
            }

            if (resultHtml) {
                element.innerHTML = resultHtml;
            } else {
                element.textContent = text;
            }

        } catch (err) {
            console.warn('Math rendering fallback:', err);
            element.textContent = text;
        }
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

    // Clear Prompt Button
    if (clearPromptBtn) {
        clearPromptBtn.addEventListener('click', () => {
            if (promptInput) {
                promptInput.value = '';
                promptInput.focus();
            }
        });
    }

    // Clear Graph Button
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

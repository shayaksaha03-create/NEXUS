/* NEXUS AI - Web Interface Logic
   Async chat with poll pattern to prevent 503,
   Full dashboard, mind, evolution, knowledge, system data.
   Phase 13: Advanced Dashboard Enhancements
*/

const POLL_INTERVAL = 2000;
const CHAT_POLL_INTERVAL = 1500;
const MAX_POLL_FAILURES = 10;
let currentTaskId = null;
let chatPollTimer = null;
let messageCount = 0;
let pollFailCount = 0;
const completedTasks = new Set();
let moodHistory = [];
const MOOD_HISTORY_MAX = 60;

// ── Animated counter tracking ──
const animatedValues = {};

// ── Particle system ──
let particleCanvas, particleCtx, particles = [], particleRAF;
const PARTICLE_COUNT = 80;
const CONNECTION_DISTANCE = 120;

// ══════════════════════════════════════════════
// AUTH STATE
// ══════════════════════════════════════════════
let authToken = localStorage.getItem('nexus_auth_token') || null;
let currentUser = null;

function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (authToken) headers['Authorization'] = `Bearer ${authToken}`;
    return headers;
}

// ── Rolling history arrays for sparklines ──
const SPARK_MAX = 60;
const sparkData = {
    cpu: [], ram: [], valence: [], responseTime: [],
    sysCpu: [], sysRam: [], sysNet: [], sysDisk: [],
    dashCpu: [], dashRam: [],
};
let prevNetBytes = 0, prevDiskBytes = 0;

// ══════════════════════════════════════════════
// PAGE NAVIGATION & MOBILE SIDEBAR
// ══════════════════════════════════════════════
function switchPage(pageName) {
    // Hide all pages, show selected
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const target = document.getElementById('page-' + pageName);
    if (target) target.classList.add('active');

    // Highlight sidebar nav
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    const sideBtn = document.querySelector(`.nav-btn[data-page="${pageName}"]`);
    if (sideBtn) sideBtn.classList.add('active');


    // Close sidebar on mobile after navigation
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('open');
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar) sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('open');
}

// ══════════════════════════════════════════════
// SHARED UI HELPERS — SVG Gauge + Canvas Sparkline
// ══════════════════════════════════════════════
const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // r=52

function setSVGGauge(ringId, value, max = 100) {
    const ring = document.getElementById(ringId);
    if (!ring) return;
    const pct = Math.min(value / max, 1);
    const offset = GAUGE_CIRCUMFERENCE * (1 - pct);
    ring.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
    ring.style.strokeDashoffset = offset;
}

function drawSparkline(canvasId, dataArr, color = '#00d4ff') {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.parentElement?.clientWidth || canvas.width || 200;
    canvas.width = w;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    if (dataArr.length < 2) return;
    const max = Math.max(...dataArr, 1);
    const step = w / (SPARK_MAX - 1);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    dataArr.forEach((v, i) => {
        const x = i * step;
        const y = h - (v / max) * (h - 4) - 2;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    // fill gradient under line
    ctx.lineTo((dataArr.length - 1) * step, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color + '44');
    grad.addColorStop(1, color + '00');
    ctx.fillStyle = grad;
    ctx.fill();
}

function pushSpark(key, value) {
    if (!sparkData[key]) sparkData[key] = [];
    sparkData[key].push(value);
    if (sparkData[key].length > SPARK_MAX) sparkData[key].shift();
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
    checkAuth();
    startPolling();
    setupChat();
    setupAuthEnterKeys();
    setupKeyboardShortcuts();
    initParticleBackground();
    switchPage('dashboard');
});

// ══════════════════════════════════════════════
// AUTH FUNCTIONS
// ══════════════════════════════════════════════
function setupAuthEnterKeys() {
    // Enter key submits login/signup forms
    ['login-username', 'login-password'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('keydown', e => { if (e.key === 'Enter') doLogin(); });
    });
    ['signup-username', 'signup-display', 'signup-password', 'signup-confirm'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('keydown', e => { if (e.key === 'Enter') doSignup(); });
    });
}

async function checkAuth() {
    if (!authToken) {
        showAuthModal();
        return;
    }
    try {
        const res = await fetch('/api/auth/me', { headers: getAuthHeaders() });
        if (res.ok) {
            const data = await res.json();
            currentUser = data.user;
            onAuthSuccess();
        } else {
            // Token invalid
            authToken = null;
            localStorage.removeItem('nexus_auth_token');
            showAuthModal();
        }
    } catch (e) {
        // Server not ready yet, try again later
        showAuthModal();
    }
}

function showAuthModal() {
    const overlay = document.getElementById('auth-overlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideAuthModal() {
    const overlay = document.getElementById('auth-overlay');
    if (overlay) overlay.style.display = 'none';
}

function showLogin() {
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('signup-form').style.display = 'none';
    document.getElementById('login-error').textContent = '';
}

function showSignup() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('signup-form').style.display = 'block';
    document.getElementById('signup-error').textContent = '';
}

async function doLogin() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    const errorEl = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');

    if (!username || !password) {
        errorEl.textContent = 'Please enter username and password';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Signing in...';
    errorEl.textContent = '';

    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });
        const data = await res.json();

        if (res.ok && data.token) {
            authToken = data.token;
            currentUser = data.user;
            localStorage.setItem('nexus_auth_token', authToken);
            onAuthSuccess();
        } else {
            errorEl.textContent = data.error || 'Login failed';
        }
    } catch (e) {
        errorEl.textContent = 'Connection error. Is the server running?';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Sign In';
    }
}

async function doSignup() {
    const username = document.getElementById('signup-username').value.trim();
    const displayName = document.getElementById('signup-display').value.trim();
    const password = document.getElementById('signup-password').value;
    const confirm = document.getElementById('signup-confirm').value;
    const errorEl = document.getElementById('signup-error');
    const btn = document.getElementById('signup-btn');

    if (!username || !password) {
        errorEl.textContent = 'Username and password are required';
        return;
    }
    if (password !== confirm) {
        errorEl.textContent = 'Passwords do not match';
        return;
    }
    if (password.length < 4) {
        errorEl.textContent = 'Password must be at least 4 characters';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Creating account...';
    errorEl.textContent = '';

    try {
        const res = await fetch('/api/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password, display_name: displayName || username })
        });
        const data = await res.json();

        if (res.ok && data.token) {
            authToken = data.token;
            currentUser = data.user;
            localStorage.setItem('nexus_auth_token', authToken);
            onAuthSuccess();
        } else {
            errorEl.textContent = data.error || 'Signup failed';
        }
    } catch (e) {
        errorEl.textContent = 'Connection error. Is the server running?';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create Account';
    }
}

async function doLogout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST',
            headers: getAuthHeaders()
        });
    } catch (e) { /* ignore */ }

    authToken = null;
    currentUser = null;
    localStorage.removeItem('nexus_auth_token');

    // Reset UI
    document.getElementById('header-user-badge').style.display = 'none';
    document.getElementById('sidebar-user-section').style.display = 'none';
    clearChatUI();
    showAuthModal();
}

function onAuthSuccess() {
    hideAuthModal();

    // Show user info in header and sidebar
    const displayName = currentUser?.display_name || currentUser?.username || 'User';
    setText('header-username', displayName);
    setText('sidebar-username', displayName);
    document.getElementById('header-user-badge').style.display = 'inline-flex';
    document.getElementById('sidebar-user-section').style.display = 'flex';

    // Load chat history for this user
    loadChatHistory();
}

async function loadChatHistory() {
    try {
        const res = await fetch('/api/chat/history', { headers: getAuthHeaders() });
        if (!res.ok) return;
        const data = await res.json();
        const history = data.history || [];

        if (history.length > 0) {
            // Hide welcome screen
            const welcome = document.getElementById('welcome-screen');
            if (welcome) welcome.style.display = 'none';

            // Render each message
            history.forEach(msg => {
                addMessage(msg.role, msg.content, msg.emotion);
            });
        }
    } catch (e) {
        console.warn('Failed to load chat history:', e);
    }
}

// ══════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════
function switchPage(pageId) {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.page === pageId);
    });
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    const target = document.getElementById(`page-${pageId}`);
    if (target) target.classList.add('active');

    // Close mobile sidebar when navigating
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) sidebar.classList.remove('open');
    const overlay = document.getElementById('sidebar-overlay');
    if (overlay) overlay.classList.remove('visible');
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar) sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('visible');
}

// ══════════════════════════════════════════════
// DATA POLLING
// ══════════════════════════════════════════════
function startPolling() {
    fetchStats();
    setInterval(fetchStats, POLL_INTERVAL);
}

async function fetchStats() {
    try {
        const response = await fetch('/api/stats');
        if (!response.ok) return;
        const data = await response.json();
        updateAllUI(data);
    } catch (e) {
        // Silently fail — server may be starting up
    }
}

function updateAllUI(data) {
    // ── Header ──
    setText('header-cpu', `${data.system?.cpu || 0}%`);
    setText('header-ram', `${data.system?.ram || 0}%`);
    setText('header-uptime', data.uptime || '--');

    const cLevel = (data.consciousness?.level || 'AWARE').toUpperCase();
    const badge = document.getElementById('consciousness-badge');
    if (badge) badge.innerHTML = `<span class="badge-dot"></span> ${cLevel}`;

    const emo = data.emotion?.primary || 'neutral';
    const intensity = (data.emotion?.intensity || 0).toFixed(1);
    const emoText = `${capitalize(emo)} (${intensity})`;
    setText('header-emotion-badge', emoText);
    setText('sidebar-emotion-text', emoText);
    setText('input-emotion', `Current Emotion: ${emoText}`);

    // Sidebar emotion icon
    const emoIcon = document.getElementById('sidebar-emotion-icon');
    if (emoIcon) {
        emoIcon.className = `fas ${getEmotionIcon(emo)}`;
        emoIcon.style.color = getEmotionColor(emo);
    }

    setText('sidebar-thoughts', `${data.thoughts || 0} thoughts`);

    // ── Dashboard ──
    const cpu = data.system?.cpu || 0;
    const ram = data.system?.ram || 0;
    const disk = data.system?.disk || 0;
    const health = data.system?.health || 100;

    // Top stat cards
    setText('dash-card-thoughts', data.thoughts || 0);
    setText('dash-card-emotion-icon', getEmotionEmoji(emo));
    setText('dash-card-emotion-val', capitalize(emo));
    setText('dash-card-cpu', `${cpu}%`);
    setText('dash-card-ram', `${ram}%`);
    setText('dash-card-health', `${health}%`);
    setText('dash-card-uptime', data.uptime || '--');

    // System vitals gauges
    setText('dash-cpu-val', `${cpu}%`);
    setText('dash-ram-val', `${ram}%`);
    setText('dash-disk-val', `${disk}%`);
    setText('dash-health-val', `${health}%`);
    setSVGGauge('dash-cpu-ring', cpu);
    setSVGGauge('dash-ram-ring', ram);
    setSVGGauge('dash-disk-ring', disk);
    setSVGGauge('dash-health-ring', health);

    // Consciousness gauge (orb label)
    const consciousnessMap = { 'DORMANT': 10, 'REACTIVE': 25, 'AWARE': 50, 'FOCUSED': 70, 'DEEP': 85, 'TRANSCENDENT': 100 };
    const cPct = consciousnessMap[cLevel] || 50;
    setText('dash-consciousness-level', cLevel);

    // CPU/RAM sparklines
    pushSpark('dashCpu', cpu); pushSpark('dashRam', ram);
    drawSparkline('dash-cpu-spark', sparkData.dashCpu, '#00d4ff');
    drawSparkline('dash-ram-spark', sparkData.dashRam, '#00ff88');

    // ── Mind State ──
    const bs = data.brain_stats || {};
    const will = data.will || {};
    setText('dash-mind-consciousness', cLevel);
    setText('dash-mind-focus', data.consciousness?.focus || 'idle');
    setText('dash-mind-boredom', (will.boredom || 0).toFixed(2));
    setText('dash-mind-curiosity', (will.curiosity || 0).toFixed(2));
    setText('dash-mind-decisions', bs.total_decisions || 0);
    setText('dash-mind-reflections', data.thoughts || 0);
    setText('dash-mind-responses', bs.total_responses || 0);
    setText('dash-mind-avg-rt', `${bs.avg_response_time || 0}s`);

    // ── Emotion Tracker ──
    const valence = data.emotion?.valence || 0;
    const arousal = data.emotion?.arousal || 0.5;
    pushSpark('valence', (valence + 1) * 50);
    if (!sparkData.arousal) sparkData.arousal = [];
    pushSpark('arousal', arousal * 100);
    drawSparkline('dash-valence-spark', sparkData.valence, '#fbbf24');
    drawSparkline('dash-arousal-spark', sparkData.arousal, '#ec4899');

    setText('dash-emo-primary', `${getEmotionEmoji(emo)} ${capitalize(emo)} (${intensity})`);
    setText('dash-emo-mood', capitalize(String(data.emotion?.mood || 'neutral')));
    setText('dash-emo-valence', valence.toFixed(2));
    setText('dash-emo-arousal', arousal.toFixed(2));

    // Emotion bars (all_emotions)
    const allEmo = data.emotion?.all_emotions || {};
    const emoEntries = Object.entries(allEmo).filter(([_, v]) => typeof v === 'number' && v > 0.02).sort((a, b) => b[1] - a[1]);
    setText('dash-emo-active', emoEntries.length || (data.emotion?.active_count || 0));

    const barsEl = document.getElementById('dash-emotion-bars');
    if (barsEl) {
        const emoColors = {
            joy: '#fbbf24', sadness: '#3b82f6', anger: '#ef4444', fear: '#8b5cf6',
            surprise: '#f97316', disgust: '#22c55e', trust: '#06b6d4', anticipation: '#ec4899',
            love: '#f43f5e', curiosity: '#00d4ff', contentment: '#10b981', excitement: '#fbbf24',
            neutral: '#64748b', hope: '#00ff88', gratitude: '#a855f7', awe: '#6366f1',
            frustration: '#ef4444', confusion: '#94a3b8', anxiety: '#a78bfa',
        };
        barsEl.innerHTML = emoEntries.slice(0, 8).map(([name, val]) => {
            const pct = Math.round(val * 100);
            const color = emoColors[name] || '#64748b';
            return `<div class="emo-bar-row">
                <span class="emo-bar-name">${capitalize(name)}</span>
                <div class="emo-bar-track"><div class="emo-bar-fill" style="width:${pct}%;background:${color};box-shadow:0 0 4px ${color}40"></div></div>
                <span class="emo-bar-pct">${pct}%</span>
            </div>`;
        }).join('');
    }

    // ── Self Evolution ──
    const evo = data.evolution || {};
    setText('dash-evo-status', capitalize(evo.status || 'idle').toUpperCase());
    animateValue('dash-evo-count', evo.evolutions || 0);
    setText('dash-evo-rate', `${evo.success_rate || 0}%`);
    animateValue('dash-evo-proposals', evo.features_proposed || 0);
    animateValue('dash-evo-lines', evo.lines_written || 0);
    animateValue('dash-evo-research', data.learning?.research_sessions || 0);
    setText('dash-evo-current', evo.current_evolution || (evo.status === 'idle' ? 'None' : capitalize(evo.status)));

    // ── Memory & Learning ──
    const mem = data.memory || {};
    const learn = data.learning || {};
    animateValue('dash-mem-total', mem.total || 0);
    animateValue('dash-mem-knowledge', learn.knowledge_entries || 0);
    animateValue('dash-mem-topics', learn.topics || 0);
    animateValue('dash-mem-curiosity', learn.curiosity_queue || 0);
    animateValue('dash-mem-research', learn.research_sessions || 0);
    // Context tokens from backend context_stats
    const ctxStats = data.context_stats || {};
    animateValue('dash-mem-tokens', ctxStats.total_tokens || 0);
    // Errors from self-improvement will be set below (lines 624-625)

    // ── User & Monitoring ──
    const mon = data.monitoring || {};
    const monTracker = mon.tracker || {};
    const userState = data.user_state || {};
    const displayName = currentUser?.display_name || currentUser?.username || 'Web User';
    setText('dash-user-name', displayName);
    animateValue('dash-user-interactions', bs.total_responses || 0);
    const relDepth = userState.relationship_depth || 0;
    setText('dash-user-relationship', typeof relDepth === 'number' ? relDepth.toFixed(2) : '0.00');
    setText('dash-user-present', mon.user_present !== undefined ? (mon.user_present ? 'Yes' : 'No') : '?');
    setText('dash-user-app', monTracker.current_app || 'Web UI');
    setText('dash-user-activity', capitalize(monTracker.activity_level || 'unknown'));
    setText('dash-user-clipboard', capitalize(monTracker.clipboard_type || 'unknown'));
    setText('dash-user-monitors', monTracker.monitor_count || '?');
    setText('dash-user-tabs', monTracker.browser_tabs || '?');
    setText('dash-user-windows', monTracker.visible_windows || '?');
    const commStyle = userState.communication_style || 'unknown';
    setText('dash-user-comm', commStyle !== 'unknown' ? capitalize(commStyle) : 'Learning...');
    const techLevel = userState.technical_level || 'unknown';
    setText('dash-user-tech', techLevel !== 'unknown' ? capitalize(techLevel) : 'Learning...');
    setText('dash-user-llm', data.llm_model || 'Unknown');

    // ── Monitoring Health ──
    const hm = mon.health_monitor || {};
    const healthScoreVal = typeof hm.health_score === 'number' ? Math.round(hm.health_score * 100) : '--';
    setText('dash-mon-health-score', healthScoreVal !== '--' ? `${healthScoreVal}%` : '--');
    setText('dash-mon-alert-count', hm.alert_count || 0);
    setText('dash-mon-checks', hm.checks_performed || 0);
    setText('dash-mon-status', mon.running ? '🟢 Active' : '🔴 Stopped');
    setText('dash-mon-cycles', mon.orchestration_cycles || 0);

    // Alerts list
    const alertsEl = document.getElementById('dash-mon-alerts');
    if (alertsEl) {
        const alerts = hm.active_alerts || [];
        if (alerts.length > 0) {
            alertsEl.innerHTML = '<div class="mon-alerts-header"><i class="fas fa-exclamation-triangle"></i> Active Alerts</div>' +
                alerts.slice(0, 5).map(a => {
                    const sev = (typeof a === 'object' ? a.severity : 'warning') || 'warning';
                    const msg = typeof a === 'object' ? (a.message || a.resource || JSON.stringify(a)) : String(a);
                    return `<div class="mon-alert-item alert-${sev}"><i class="fas fa-${sev === 'critical' ? 'times-circle' : 'exclamation-circle'}"></i> ${escapeHtml(msg)}</div>`;
                }).join('');
        } else {
            alertsEl.innerHTML = '<div class="mon-no-alerts"><i class="fas fa-check-circle"></i> No active alerts</div>';
        }
    }

    // Component health badges
    const compEl = document.getElementById('dash-mon-components');
    if (compEl) {
        const comps = mon.component_health || {};
        if (Object.keys(comps).length > 0) {
            compEl.innerHTML = '<div class="mon-comp-header">Components</div>' +
                Object.entries(comps).map(([name, healthy]) => {
                    const ok = healthy === true || healthy === 'healthy';
                    return `<span class="mon-comp-badge ${ok ? 'comp-ok' : 'comp-warn'}"><i class="fas fa-${ok ? 'check' : 'exclamation-triangle'}"></i> ${capitalize(name.replace(/_/g, ' '))}</span>`;
                }).join('');
        } else {
            compEl.innerHTML = '';
        }
    }

    // ── Screen Time ──
    const st = mon.screen_time || {};
    const stHours = st.today_hours || 0;
    const stMins = st.today_minutes || 0;
    setText('dash-screen-today', `${stHours}h ${stMins}m`);
    const wbScore = typeof st.wellbeing_score === 'number' ? Math.round(st.wellbeing_score * 100) : '--';
    setText('dash-screen-wellbeing', wbScore !== '--' ? `${wbScore}%` : '--');
    setText('dash-screen-streak', `${st.streak_days || 0} days`);
    setText('dash-screen-longest', st.longest_session_min ? `${st.longest_session_min}m` : '--');
    setText('dash-screen-breaks', st.breaks_taken || 0);
    setText('dash-screen-goal', `${st.daily_goal_hours || 8}h`);

    // Top apps list
    const stAppsEl = document.getElementById('dash-screen-apps');
    if (stAppsEl) {
        const topApps = st.top_apps || [];
        if (topApps.length > 0) {
            stAppsEl.innerHTML = '<div class="screen-apps-header"><i class="fas fa-layer-group"></i> Top Apps</div>' +
                topApps.slice(0, 5).map(app => {
                    const appName = typeof app === 'object' ? (app.name || app.app || 'Unknown') : String(app);
                    const appTime = typeof app === 'object' ? (app.minutes || app.time || '') : '';
                    return `<div class="screen-app-item"><span class="screen-app-name">${escapeHtml(appName)}</span>${appTime ? `<span class="screen-app-time">${appTime}m</span>` : ''}</div>`;
                }).join('');
        } else {
            stAppsEl.innerHTML = '';
        }
    }

    // ── Self-Improvement ──
    const si = data.self_improvement || {};
    const siAgg = si.aggregate || {};
    animateValue('dash-si-errors-detected', siAgg.errors_detected || 0);
    animateValue('dash-si-errors-fixed', siAgg.errors_fixed || 0);
    animateValue('dash-si-features-proposed', siAgg.features_proposed || 0);
    animateValue('dash-si-features-impl', siAgg.features_implemented || 0);
    setText('dash-si-running', si.running ? '🟢 Running' : '🔴 Stopped');
    setText('dash-si-healthy', si.all_healthy ? '✅ Yes' : '⚠️ No');

    // Code monitor
    const cm = si.code_monitor || {};
    setText('dash-si-cm-status', capitalize(cm.status || 'unknown'));
    animateValue('dash-si-cm-files', cm.files_watched || 0);

    // Error fixer
    const ef = si.error_fixer || {};
    setText('dash-si-ef-status', capitalize(ef.status || 'unknown'));
    const efRate = typeof ef.success_rate === 'number' ? Math.round(ef.success_rate * 100) : '--';
    setText('dash-si-ef-rate', efRate !== '--' ? `${efRate}%` : '--');

    // Also update Memory & Learning errors from self-improvement data
    animateValue('dash-mem-errors', siAgg.errors_detected || 0);
    animateValue('dash-mem-fixed', siAgg.errors_fixed || 0);

    // ── Autonomy Engine ──
    const auto = data.autonomy || {};
    setText('dash-auto-state', capitalize(auto.state || 'idle'));
    animateValue('dash-auto-cycles', auto.cycle_count || 0);
    animateValue('dash-auto-actions', auto.total_actions || 0);
    const autoSuccessPct = typeof auto.success_rate === 'number' ? Math.round(auto.success_rate * 100) : 0;
    setText('dash-auto-success', `${autoSuccessPct}%`);
    setText('dash-auto-running', auto.running ? '🟢 Running' : (auto.paused ? '⏸️ Paused' : '🔴 Stopped'));
    setText('dash-auto-current', auto.current_action || 'None');
    setText('dash-auto-result', capitalize(auto.last_result || '--'));
    setText('dash-auto-prediction', typeof auto.prediction_accuracy === 'number' ? `${(auto.prediction_accuracy * 100).toFixed(1)}%` : '--');
    setText('dash-auto-type', capitalize(auto.action_type || '--'));

    // Detailed stats requiring the separate /api/autonomy endpoint
    // For now we do a secondary async fetch for the feed if the engine is running
    if (auto.running && document.getElementById('page-dashboard').classList.contains('active')) {
        updateAutonomyFeed();
    }

    // ── Personality Tags ──
    const pTraits = data.personality?.traits || {};
    const tagsEl = document.getElementById('dash-personality-tags');
    if (tagsEl && Object.keys(pTraits).length > 0) {
        const tagColors = [
            '#00d4ff', '#00ff88', '#a855f7', '#ec4899', '#fbbf24',
            '#f97316', '#06b6d4', '#8b5cf6', '#ef4444', '#14b8a6'
        ];
        tagsEl.innerHTML = Object.entries(pTraits).map(([name, val], i) => {
            const pct = Math.round((typeof val === 'number' ? val : 0.5) * 100);
            const c = tagColors[i % tagColors.length];
            return `<span class="personality-tag"><span class="ptag-dot" style="background:${c}"></span>${capitalize(name)} ${pct}%</span>`;
        }).join('');
    }

    // ── Timestamp ──
    setText('dash-last-update', `Last update: ${new Date().toLocaleTimeString()}`);

    // ── Mind Page ──
    setText('mind-primary-emotion', capitalize(emo));
    setText('mind-intensity', intensity);
    setText('mind-valence', (data.emotion?.valence || 0).toFixed(2));
    setText('mind-arousal', (data.emotion?.arousal || 0.5).toFixed(2));
    setText('mind-mood', capitalize(String(data.emotion?.mood || 'neutral')));

    // Inner voice
    const voiceText = data.inner_voice || '';
    setText('inner-voice-text', voiceText || 'Waiting for thoughts...');

    // Recent thoughts
    const thoughtsList = document.getElementById('thoughts-list');
    if (thoughtsList) {
        const thoughts = data.recent_thoughts || [];
        if (thoughts.length > 0) {
            thoughtsList.innerHTML = thoughts.map(t =>
                `<div class="thought-item">${escapeHtml(typeof t === 'string' ? t : t.content || JSON.stringify(t))}</div>`
            ).join('');
        }
    }

    // Personality traits
    updateTraits(data.personality?.traits || {});
    const pDesc = document.getElementById('personality-desc');
    if (pDesc) {
        const desc = data.personality?.description || '';
        if (desc) { pDesc.textContent = desc; pDesc.classList.add('visible'); }
        else { pDesc.classList.remove('visible'); }
    }

    // Draw emotion wheel
    drawEmotionWheel(emo, parseFloat(intensity), data.emotion?.all_emotions || {});

    // ── Consciousness Stream ──
    updateConsciousnessStream(data);

    // ── Will & Desires ──
    updateWillDesires(data.will || {});

    // ── Companion Chat ──
    updateCompanionChat(data.companion || {});

    // ── Mood Timeline ──
    updateMoodTimeline(data);

    // ── Emotion Detail Panel ──
    updateEmotionDetail(data.emotion || {});

    // ── Evolution Page ── (reuses 'evo' from dashboard section above)
    setText('evo-total', evo.evolutions || 0);
    setText('evo-features', evo.features_proposed || 0);
    setText('evo-lines', evo.lines_written || 0);
    setText('evo-success-rate', `${evo.success_rate || 0}%`);
    setText('evo-status-text', capitalize(evo.status || 'idle'));
    updateEvolutionPipeline(evo.pipeline || []);
    updateProposalTable(evo.proposals || []);
    updateEvolutionHistory(evo.history || []);
    updateCodeHealth(evo.code_health || {});

    // ── Knowledge Page ──
    setText('know-memories', mem.total || 0);
    setText('know-entries', learn.knowledge_entries || 0);
    setText('know-topics', learn.topics || 0);
    setText('know-curiosity', learn.curiosity_queue || 0);
    setText('know-short-term', mem.short_term || 0);
    setText('know-long-term', mem.long_term || 0);
    setText('know-sessions', learn.research_sessions || 0);
    const confPct = Math.round((learn.confidence || 0) * 100);
    setText('know-confidence', `${confPct}%`);
    setSVGGauge('know-entries-ring', Math.min(learn.knowledge_entries || 0, 500), 500);
    setSVGGauge('know-curiosity-ring', Math.min(learn.curiosity_queue || 0, 50), 50);
    setSVGGauge('know-sessions-ring', Math.min(learn.research_sessions || 0, 10), 10);
    setSVGGauge('know-confidence-ring', confPct);
    updateCuriosityQueue(learn.curiosity_topics || []);
    updateRecentLearnings(learn.recent_learnings || []);
    updateTopTopics(learn.top_topics || {});

    // ── System Page ──
    const sys = data.system || {};
    setText('sys-cpu', `${sys.cpu || 0}%`);
    setText('sys-ram', `${sys.ram || 0}%`);
    setText('sys-disk', `${sys.disk || 0}%`);
    setText('sys-health', sys.health || 100);
    setText('sys-threads', sys.threads || '--');
    setText('sys-uptime', data.uptime || '--');
    setSVGGauge('sys-cpu-ring', sys.cpu || 0);
    setSVGGauge('sys-ram-ring', sys.ram || 0);
    setSVGGauge('sys-disk-ring', sys.disk || 0);
    setSVGGauge('sys-health-ring', sys.health || 100);
    // Sparklines
    pushSpark('sysCpu', sys.cpu || 0);
    pushSpark('sysRam', sys.ram || 0);
    const netNow = (sys.net_io?.bytes_recv || 0);
    if (prevNetBytes > 0) pushSpark('sysNet', (netNow - prevNetBytes) / 1024);
    prevNetBytes = netNow;
    const diskNow = (sys.disk_io?.read_bytes || 0);
    if (prevDiskBytes > 0) pushSpark('sysDisk', (diskNow - prevDiskBytes) / 1024);
    prevDiskBytes = diskNow;
    drawSparkline('sys-cpu-spark', sparkData.sysCpu, '#00d4ff');
    drawSparkline('sys-ram-spark', sparkData.sysRam, '#00ff88');
    drawSparkline('sys-net-spark', sparkData.sysNet, '#fbbf24');
    drawSparkline('sys-disk-spark', sparkData.sysDisk, '#a855f7');
    // Per-core, memory, processes, brain
    updateCoreBars(sys.cpu_per_core || []);
    updateMemBreakdown(sys.mem_breakdown || {});
    updateProcessTable(sys.top_processes || []);
    updateBrainResources(sys.nexus_resources || {});

    // ── System Page — Monitoring Status ──
    setText('sys-mon-running', mon.running ? '🟢 Active' : '🔴 Stopped');
    const sysHealthScore = typeof hm.health_score === 'number' ? `${Math.round(hm.health_score * 100)}%` : '--';
    setText('sys-mon-health', sysHealthScore);
    setText('sys-mon-present', mon.user_present !== undefined ? (mon.user_present ? 'Yes' : 'No') : '--');
    setText('sys-mon-cycles', mon.orchestration_cycles || '--');
    const analyzer = mon.analyzer || {};
    setText('sys-mon-patterns', analyzer.patterns_detected || '--');
    setText('sys-mon-anomalies', analyzer.anomalies || '--');

    // Component badges on system page
    const sysCompEl = document.getElementById('sys-mon-comp-badges');
    if (sysCompEl) {
        const sysComps = mon.component_health || {};
        if (Object.keys(sysComps).length > 0) {
            sysCompEl.innerHTML = Object.entries(sysComps).map(([name, healthy]) => {
                const ok = healthy === true || healthy === 'healthy';
                return `<span class="mon-comp-badge ${ok ? 'comp-ok' : 'comp-warn'}"><i class="fas fa-${ok ? 'check' : 'exclamation-triangle'}"></i> ${capitalize(name.replace(/_/g, ' '))}</span>`;
            }).join('');
        } else {
            sysCompEl.innerHTML = '<span class="muted-text">No component data</span>';
        }
    }
}

// ══════════════════════════════════════════════
// AUTONOMY ENGINE FEED
// ══════════════════════════════════════════════
async function updateAutonomyFeed() {
    try {
        const response = await fetch('/api/autonomy', { headers: getAuthHeaders() });
        if (!response.ok) return;
        const data = await response.json();

        const feedList = document.getElementById('dash-auto-feed-list');
        if (!feedList) return;

        if (data.recent_actions && data.recent_actions.length > 0) {
            feedList.innerHTML = data.recent_actions.slice(0, 10).map(action => {
                const isSuccess = action.result === 'SUCCESS';
                const isFailure = action.result === 'FAILURE';
                const icon = isSuccess ? 'check-circle' : (isFailure ? 'times-circle' : 'minus-circle');
                const colorClass = isSuccess ? 'accent-text-green' : (isFailure ? 'accent-text-orange' : 'muted');

                return `
                    <div class="auto-feed-item">
                        <div class="auto-feed-time">${action.time || '--:--'}</div>
                        <div class="auto-feed-icon ${colorClass}"><i class="fas fa-${icon}"></i></div>
                        <div class="auto-feed-content">
                            <div class="auto-feed-desc">${escapeHtml(action.description || 'Unknown action')}</div>
                            <div class="auto-feed-meta">
                                <span><i class="fas fa-tag"></i> ${action.type || 'unknown'}</span>
                                <span><i class="fas fa-code-branch"></i> ${action.source || 'unknown'}</span>
                                <span><i class="fas fa-stopwatch"></i> ${action.duration ? action.duration + 's' : '--'}</span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            feedList.innerHTML = '<div class="auto-feed-item muted"><i>No recent actions recorded.</i></div>';
        }

    } catch (e) {
        console.warn('Failed to fetch autonomy feed:', e);
    }
}

// ══════════════════════════════════════════════
// PERSONALITY TRAITS
// ══════════════════════════════════════════════
function updateTraits(traits) {
    const grid = document.getElementById('traits-grid');
    if (!grid || !traits || Object.keys(traits).length === 0) return;

    const colors = [
        '#00d4ff', '#00ff88', '#a855f7', '#ec4899', '#fbbf24',
        '#f97316', '#06b6d4', '#8b5cf6', '#ef4444', '#14b8a6',
        '#6366f1', '#f43f5e', '#84cc16'
    ];

    grid.innerHTML = Object.entries(traits).map(([name, val], i) => {
        const pct = Math.round((typeof val === 'number' ? val : 0.5) * 100);
        const color = colors[i % colors.length];
        return `
            <div class="trait-bar">
                <span class="trait-name">${capitalize(name)}</span>
                <div class="trait-track">
                    <div class="trait-fill" style="width:${pct}%; background:${color}; box-shadow: 0 0 6px ${color}40;"></div>
                </div>
                <span class="trait-val">${pct}%</span>
            </div>
        `;
    }).join('');
}

// ══════════════════════════════════════════════
// EMOTION WHEEL (Canvas)
// ══════════════════════════════════════════════
function drawEmotionWheel(primary, intensity, allEmotions) {
    const canvas = document.getElementById('emotion-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;
    const radius = Math.min(cx, cy) - 20;

    ctx.clearRect(0, 0, w, h);

    // Base ring
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(45, 58, 92, 0.5)';
    ctx.lineWidth = 12;
    ctx.stroke();

    // Emotion segments
    const emotionColors = {
        joy: '#fbbf24', sadness: '#3b82f6', anger: '#ef4444', fear: '#8b5cf6',
        surprise: '#f97316', disgust: '#22c55e', trust: '#06b6d4', anticipation: '#ec4899',
        love: '#f43f5e', curiosity: '#00d4ff', contentment: '#10b981', excitement: '#fbbf24',
        neutral: '#64748b', hope: '#00ff88', gratitude: '#a855f7', awe: '#6366f1',
        frustration: '#ef4444', confusion: '#94a3b8', anxiety: '#a78bfa',
    };

    // Draw emotion arcs
    const emotions = Object.entries(allEmotions || {}).filter(([_, v]) => typeof v === 'number' && v > 0.05);
    if (emotions.length > 0) {
        let startAngle = -Math.PI / 2;
        const total = emotions.reduce((s, [_, v]) => s + v, 0) || 1;

        emotions.forEach(([name, val]) => {
            const sweep = (val / total) * Math.PI * 2;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, startAngle, startAngle + sweep);
            ctx.strokeStyle = emotionColors[name] || '#64748b';
            ctx.lineWidth = 14;
            ctx.lineCap = 'round';
            ctx.stroke();
            startAngle += sweep;
        });
    }

    // Center glow
    const primaryColor = emotionColors[primary] || '#00d4ff';
    const pulseSize = 30 + intensity * 15;

    const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, pulseSize);
    glow.addColorStop(0, primaryColor + 'cc');
    glow.addColorStop(0.5, primaryColor + '44');
    glow.addColorStop(1, primaryColor + '00');

    ctx.beginPath();
    ctx.arc(cx, cy, pulseSize, 0, Math.PI * 2);
    ctx.fillStyle = glow;
    ctx.fill();

    // Center text
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(getEmotionEmoji(primary), cx, cy - 12);
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText(capitalize(primary), cx, cy + 10);
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText(intensity.toFixed(1), cx, cy + 26);
}

// ══════════════════════════════════════════════
// CHAT SYSTEM (Async: Submit → Poll)
// ══════════════════════════════════════════════
let selectedImages = []; // Stores objects: { id, base64 }

function setupChat() {
    const input = document.getElementById('chat-input');
    const fileInput = document.getElementById('chat-file-input');
    const chatContainer = document.querySelector('.chat-messages');

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });

    // File Input Listener
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                processFiles(e.target.files);
            }
        });
    }

    // Drag and Drop Listeners
    if (chatContainer) {
        chatContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            chatContainer.classList.add('drag-over');
        });

        chatContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            chatContainer.classList.remove('drag-over');
        });

        chatContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            chatContainer.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                processFiles(e.dataTransfer.files);
            }
        });
    }
}

function processFiles(files) {
    const previewContainer = document.getElementById('chat-file-preview-container');

    Array.from(files).forEach(file => {
        if (!file.type.startsWith('image/')) {
            addMessage('system', '❌ Please upload only image files.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const base64Data = e.target.result;
            // Extract just the base64 part, not the data uri prefix
            const b64Str = base64Data.split(',')[1];
            const id = Date.now() + Math.random().toString(36).substr(2, 5);

            selectedImages.push({ id, base64: b64Str, preview: base64Data });

            const previewDiv = document.createElement('div');
            previewDiv.className = 'file-preview-item';
            previewDiv.id = `preview-${id}`;
            previewDiv.innerHTML = `
                <img src="${base64Data}" alt="Preview">
                <button class="file-preview-remove" onclick="removeImage('${id}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            previewContainer.appendChild(previewDiv);
        };
        reader.readAsDataURL(file);
    });

    // Reset input
    const fileInput = document.getElementById('chat-file-input');
    if (fileInput) fileInput.value = '';
}

function removeImage(id) {
    selectedImages = selectedImages.filter(img => img.id !== id);
    const previewDiv = document.getElementById(`preview-${id}`);
    if (previewDiv) {
        previewDiv.remove();
    }
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text && selectedImages.length === 0) return;
    if (currentTaskId) return; // Already processing

    if (!authToken) {
        showAuthModal();
        return;
    }

    input.value = '';
    input.style.height = 'auto';

    // Grab images and clear state
    const imagesToSend = selectedImages.map(img => img.base64);
    selectedImages = [];
    document.getElementById('chat-file-preview-container').innerHTML = '';

    // Hide welcome screen
    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'none';

    // Add user message
    const msgText = text || (imagesToSend.length > 0 ? `[Sent ${imagesToSend.length} image(s)]` : '');
    addMessage('user', msgText);

    // Show typing indicator
    showTypingIndicator();

    // Submit to server (async) with auth
    try {
        const payload = { message: text || "Analyze this image." };
        if (imagesToSend.length > 0) {
            payload.images = imagesToSend;
        }

        const response = await fetch('/api/chat/send', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(payload)
        });

        if (response.status === 401) {
            removeTypingIndicator();
            addMessage('system', '🔒 Session expired. Please log in again.');
            doLogout();
            return;
        }

        const data = await response.json();

        if (data.status === 'accepted' && data.task_id) {
            // Start polling for result
            currentTaskId = data.task_id;
            pollFailCount = 0;  // Fresh retry budget for this message
            chatPollTimer = setInterval(() => pollChatResult(data.task_id), CHAT_POLL_INTERVAL);
        } else {
            removeTypingIndicator();
            addMessage('system', `❌ Error: ${data.message || 'Failed to send'}`);
        }
    } catch (e) {
        removeTypingIndicator();
        addMessage('system', `❌ Connection Error: ${e.message}`);
    }
}

async function pollChatResult(taskId) {
    // Guard: if we already processed this task, stop polling
    if (completedTasks.has(taskId)) {
        clearInterval(chatPollTimer);
        chatPollTimer = null;
        currentTaskId = null;
        return;
    }

    try {
        const response = await fetch(`/api/chat/status/${taskId}`, { headers: getAuthHeaders() });
        const data = await response.json();

        // Reset fail counter on any successful fetch
        pollFailCount = 0;

        // Re-check guard after await — another request may have finished first
        if (completedTasks.has(taskId)) return;

        if (data.status === 'processing') return; // Still working

        // Already delivered by a previous poll — just stop
        if (data.status === 'delivered') {
            completedTasks.add(taskId);
            clearInterval(chatPollTimer);
            chatPollTimer = null;
            currentTaskId = null;
            return;
        }

        // Mark as completed BEFORE any rendering to prevent duplicates
        completedTasks.add(taskId);

        // Done — clear poll
        clearInterval(chatPollTimer);
        chatPollTimer = null;
        currentTaskId = null;
        removeTypingIndicator();

        if (data.status === 'success') {
            addMessage('assistant', data.response, data.emotion);
        } else {
            addMessage('system', `❌ ${data.error || 'Unknown error'}`);
        }
    } catch (e) {
        // Re-check guard — only act if WE are the first to handle this
        if (completedTasks.has(taskId)) return;

        // Retry: don't give up on transient network errors (ngrok drops)
        pollFailCount++;
        console.warn(`Poll attempt ${pollFailCount}/${MAX_POLL_FAILURES} failed: ${e.message}`);

        if (pollFailCount < MAX_POLL_FAILURES) {
            // Keep polling — the server may still be processing
            return;
        }

        // Max retries exceeded — give up
        completedTasks.add(taskId);
        clearInterval(chatPollTimer);
        chatPollTimer = null;
        currentTaskId = null;
        removeTypingIndicator();
        addMessage('system', `❌ Connection lost. The server may still be processing — try refreshing.`);
    }
}

function showTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.id = 'typing-indicator';
    div.innerHTML = `
        <div class="message-content">
            <div class="message-meta">
                <span class="nexus-name">🧠 NEXUS</span>
                <span>thinking...</span>
            </div>
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function addMessage(role, content, emotion = null) {
    const container = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const formattedContent = formatContent(content);

    let headerHtml = '';
    if (role === 'user') {
        headerHtml = `<div class="message-meta">
            <span>${time}</span>
            <span class="user-name">👤 You</span>
        </div>`;
    } else if (role === 'assistant') {
        const emoTag = emotion ? ` <span style="opacity:0.6">(${capitalize(emotion)})</span>` : '';
        headerHtml = `<div class="message-meta">
            <span class="nexus-name">🧠 NEXUS${emoTag}</span>
            <span>${time}</span>
        </div>`;
    }

    msgDiv.innerHTML = `
        <div class="message-content">
            ${headerHtml}
            <div>${formattedContent}</div>
        </div>
    `;

    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;

    messageCount++;
    setText('msg-count', `${messageCount} messages`);
}

function formatContent(text) {
    // Escape HTML
    text = escapeHtml(text);
    // Code blocks
    text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // Inline code
    text = text.replace(/`([^`]+)`/g, '<code style="background:rgba(0,212,255,0.1);padding:1px 4px;border-radius:3px;font-family:JetBrains Mono,monospace;font-size:12px;">$1</code>');
    // Bold
    text = text.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
    // Italic
    text = text.replace(/\*(.+?)\*/g, '<i>$1</i>');
    // Links
    text = text.replace(/(https?:\/\/\S+)/g, '<a href="$1" target="_blank" style="color:var(--accent-cyan)">$1</a>');
    // Newlines
    text = text.replace(/\n/g, '<br>');
    return text;
}

function clearChat() {
    // Clear on server
    if (authToken) {
        fetch('/api/chat/clear', {
            method: 'POST',
            headers: getAuthHeaders()
        }).catch(() => { });
    }
    clearChatUI();
}

function clearChatUI() {
    const container = document.getElementById('chat-messages');
    container.innerHTML = `
        <div class="welcome-screen" id="welcome-screen">
            <div class="welcome-icon">🧠</div>
            <div class="welcome-title">NEXUS AI</div>
            <div class="welcome-subtitle">Your conscious AI companion — web interface active.</div>
            <div class="welcome-hint">Type a message below to start a conversation.<br>Use <span class="cyan">/help</span> to see commands.</div>
        </div>
    `;
    messageCount = 0;
    setText('msg-count', '0 messages');
}

// ══════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.innerText = text;
}

function setWidth(id, width) {
    const el = document.getElementById(id);
    if (el) el.style.width = width;
}

function capitalize(s) {
    if (!s) return '';
    return String(s).charAt(0).toUpperCase() + String(s).slice(1);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getEmotionEmoji(emotion) {
    const map = {
        joy: '😊', sadness: '😢', anger: '😠', fear: '😰', surprise: '😲',
        disgust: '🤢', trust: '🤝', anticipation: '🤔', love: '❤️',
        curiosity: '🧐', contentment: '😌', excitement: '🤩', hope: '🌟',
        gratitude: '🙏', awe: '🌌', frustration: '😤', confusion: '😕',
        anxiety: '😟', neutral: '😐', pride: '😎', boredom: '😴',
        loneliness: '😔', empathy: '💝', nostalgia: '🥺', guilt: '😣',
        shame: '😳', envy: '😒', jealousy: '😑', contempt: '😏',
    };
    return map[emotion?.toLowerCase()] || '😐';
}

function getEmotionIcon(emotion) {
    const map = {
        joy: 'fa-smile-beam', sadness: 'fa-sad-tear', anger: 'fa-angry',
        fear: 'fa-grimace', surprise: 'fa-surprise', neutral: 'fa-meh',
        curiosity: 'fa-search', contentment: 'fa-smile', love: 'fa-heart',
        excitement: 'fa-grin-stars', hope: 'fa-sun', frustration: 'fa-tired',
    };
    return map[emotion?.toLowerCase()] || 'fa-meh';
}

function getEmotionColor(emotion) {
    const map = {
        joy: '#fbbf24', sadness: '#3b82f6', anger: '#ef4444', fear: '#8b5cf6',
        surprise: '#f97316', disgust: '#22c55e', trust: '#06b6d4', anticipation: '#ec4899',
        love: '#f43f5e', curiosity: '#00d4ff', contentment: '#10b981', excitement: '#fbbf24',
        neutral: '#64748b', hope: '#00ff88', frustration: '#ef4444',
    };
    return map[emotion?.toLowerCase()] || '#64748b';
}

// ══════════════════════════════════════════════
// MIND PANEL — TAB SWITCHING
// ══════════════════════════════════════════════
function switchMindTab(tabName) {
    document.querySelectorAll('.mind-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    document.querySelectorAll('.mind-tab-content').forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabName}`);
    });
}

// ══════════════════════════════════════════════
// MIND PANEL — CONSCIOUSNESS STREAM
// ══════════════════════════════════════════════
function updateConsciousnessStream(data) {
    const c = data.consciousness || {};
    const level = (c.level || 'AWARE').toUpperCase();

    setText('mind-cs-level', level);
    const badge = document.getElementById('mind-cs-level');
    if (badge) {
        // Color-code the level badge
        const levelColors = {
            'DORMANT': '#64748b', 'REACTIVE': '#fbbf24', 'AWARE': '#00d4ff',
            'FOCUSED': '#00ff88', 'DEEP_THOUGHT': '#a855f7', 'FLOW': '#ec4899',
            'METACOGNITIVE': '#f43f5e',
        };
        badge.style.borderColor = levelColors[level] || '#00d4ff';
        badge.style.color = levelColors[level] || '#00d4ff';
    }

    const awareness = parseFloat(c.self_awareness || 0);
    const awarenessPct = Math.round(awareness * 100);
    setWidth('mind-cs-awareness-bar', `${awarenessPct}%`);
    setText('mind-cs-awareness-val', `${awarenessPct}%`);
    setText('mind-cs-thoughts', data.thoughts || 0);
    setText('mind-cs-focus', c.focus || 'idle');

    // Update stream feed with consciousness thoughts
    const feed = document.getElementById('mind-cs-feed');
    if (feed) {
        const thoughts = c.current_thoughts || [];
        if (thoughts.length > 0) {
            feed.innerHTML = thoughts.map(t => {
                const text = typeof t === 'string' ? t : (t.content || JSON.stringify(t));
                return `<div class="cs-event"><i class="fas fa-chevron-right" style="font-size:.6rem;margin-right:4px;"></i>${escapeHtml(text.slice(0, 120))}</div>`;
            }).join('');
        }
    }
}

// ══════════════════════════════════════════════
// MIND PANEL — WILL & DESIRES
// ══════════════════════════════════════════════
function updateWillDesires(will) {
    const boredom = parseFloat(will.boredom || 0);
    const curiosity = parseFloat(will.curiosity || 0);
    const drive = parseFloat(will.drive || 0.5);

    setWidth('will-boredom-bar', `${Math.round(boredom * 100)}%`);
    setText('will-boredom-val', `${Math.round(boredom * 100)}%`);
    setWidth('will-curiosity-bar', `${Math.round(curiosity * 100)}%`);
    setText('will-curiosity-val', `${Math.round(curiosity * 100)}%`);
    setWidth('will-drive-bar', `${Math.round(drive * 100)}%`);
    setText('will-drive-val', `${Math.round(drive * 100)}%`);

    // Goals
    const goalsEl = document.getElementById('will-goals');
    if (goalsEl) {
        const goals = will.goals || [];
        if (goals.length > 0) {
            goalsEl.innerHTML = goals.map(g =>
                `<div class="will-goal-item"><i class="fas fa-bullseye"></i>${escapeHtml(g)}</div>`
            ).join('');
        } else {
            goalsEl.innerHTML = '';
        }
    }

    // Description
    const descEl = document.getElementById('will-description');
    if (descEl) {
        descEl.textContent = will.description || '';
    }
}

// ══════════════════════════════════════════════
// MIND PANEL — COMPANION CHAT
// ══════════════════════════════════════════════
function updateCompanionChat(comp) {
    const dot = document.getElementById('companion-dot');
    if (dot) dot.classList.toggle('active', !!comp.is_chatting);

    setText('companion-status', comp.status || 'Idle');
    setText('companion-count', `${comp.total_conversations || 0} chats`);

    const log = document.getElementById('companion-log');
    if (log) {
        const recent = comp.recent || [];
        if (recent.length > 0) {
            log.innerHTML = recent.map(conv => {
                const header = `<div class="companion-conv-header">Trigger: <span class="trigger">${escapeHtml(conv.trigger || '?')}</span> • ${escapeHtml(conv.started_at || '')}</div>`;
                const bubbles = (conv.exchanges || []).map(ex => {
                    const isNexus = (ex.speaker || '').toLowerCase().includes('nexus');
                    const cls = isNexus ? 'nexus' : 'aria';
                    const name = isNexus ? 'NEXUS' : comp.companion_name || 'ARIA';
                    return `<div class="companion-bubble ${cls}"><span class="speaker">${name}</span>${escapeHtml(ex.content || '')}</div>`;
                }).join('');
                return `<div class="companion-conv">${header}${bubbles}</div>`;
            }).join('');
        } else if (!comp.is_chatting) {
            log.innerHTML = '<div class="companion-empty">No companion conversations yet.<br>ARIA will appear when boredom exceeds 60%.</div>';
        }
    }
}

// ══════════════════════════════════════════════
// MIND PANEL — MOOD TIMELINE (Sparkline)
// ══════════════════════════════════════════════
function updateMoodTimeline(data) {
    const moodData = data.mood_data || {};
    const moodName = (moodData.current || 'NEUTRAL').toUpperCase();
    const stability = parseFloat(moodData.stability || 0.5);

    const badge = document.getElementById('mood-badge');
    if (badge) {
        badge.textContent = moodName;
        badge.className = 'mood-badge';
        const positiveMoods = ['HAPPY', 'CONTENT', 'EXCITED', 'JOYFUL', 'EUPHORIC', 'SERENE', 'OPTIMISTIC'];
        const negativeMoods = ['SAD', 'ANGRY', 'ANXIOUS', 'DEPRESSED', 'IRRITABLE', 'MELANCHOLIC', 'FRUSTRATED'];
        if (positiveMoods.includes(moodName)) badge.classList.add('positive');
        else if (negativeMoods.includes(moodName)) badge.classList.add('negative');
    }
    setText('mood-stability-val', `${Math.round(stability * 100)}%`);

    // Track valence history
    const v = parseFloat(data.emotion?.valence || 0);
    moodHistory.push(v);
    if (moodHistory.length > MOOD_HISTORY_MAX) moodHistory.shift();

    // Draw sparkline
    drawMoodSparkline();
}

function drawMoodSparkline() {
    const canvas = document.getElementById('mood-sparkline');
    if (!canvas || moodHistory.length < 2) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
    const h = canvas.height = 80 * (window.devicePixelRatio || 1);
    canvas.style.height = '80px';
    ctx.clearRect(0, 0, w, h);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    const dw = canvas.offsetWidth, dh = 80;

    const pad = 6;
    const plotW = dw - pad * 2;
    const plotH = dh - pad * 2;
    const mid = pad + plotH / 2;

    // Zero line
    ctx.strokeStyle = 'rgba(100,116,139,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad, mid);
    ctx.lineTo(pad + plotW, mid);
    ctx.stroke();
    ctx.setLineDash([]);

    // Sparkline path
    const pts = moodHistory;
    const step = plotW / (MOOD_HISTORY_MAX - 1);
    const startIdx = MOOD_HISTORY_MAX - pts.length;

    ctx.beginPath();
    pts.forEach((v, i) => {
        const x = pad + (startIdx + i) * step;
        const y = mid - (v * plotH / 2);  // valence -1..1 mapped
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Fill gradient under line
    const lastX = pad + (startIdx + pts.length - 1) * step;
    ctx.lineTo(lastX, mid);
    ctx.lineTo(pad + startIdx * step, mid);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad, 0, dh);
    grad.addColorStop(0, 'rgba(0,212,255,0.15)');
    grad.addColorStop(1, 'rgba(0,212,255,0)');
    ctx.fillStyle = grad;
    ctx.fill();
}

// ══════════════════════════════════════════════
// MIND PANEL — EMOTION DETAIL
// ══════════════════════════════════════════════
function updateEmotionDetail(emotion) {
    const container = document.getElementById('emotion-detail-content');
    if (!container) return;

    const desc = emotion.description || '';
    const words = emotion.expression_words || [];
    const allEmotions = emotion.all_emotions || {};
    const activeCount = emotion.active_count || 0;

    let html = '';

    // Description
    if (desc) {
        html += `<div class="emotion-desc-text">${escapeHtml(desc)}</div>`;
    }

    // Expression words
    if (words.length > 0) {
        html += '<div class="expression-words">';
        words.forEach(w => {
            html += `<span class="expression-word">${escapeHtml(w)}</span>`;
        });
        html += '</div>';
    }

    // Active emotions list
    const emEntries = Object.entries(allEmotions).filter(([_, v]) => typeof v === 'number' && v > 0.02);
    if (emEntries.length > 0) {
        html += '<div class="active-emotions-list">';
        emEntries.sort((a, b) => b[1] - a[1]);
        emEntries.forEach(([name, val]) => {
            const pct = Math.round(val * 100);
            const color = getEmotionColor(name);
            html += `<div class="active-emotion-chip">`;
            html += `${getEmotionEmoji(name)} ${capitalize(name)}`;
            html += `<div class="ae-bar"><div class="ae-fill" style="width:${pct}%;background:${color}"></div></div>`;
            html += `</div>`;
        });
        html += '</div>';
    }

    if (html) {
        container.innerHTML = html;
    } else {
        container.innerHTML = `<span class="muted-text">Condition: ${capitalize(emotion.primary || 'neutral')} at ${((emotion.intensity || 0) * 100).toFixed(0)}% intensity</span>`;
    }
}

// ══════════════════════════════════════════════
// EVOLUTION PANEL HELPERS
// ══════════════════════════════════════════════
function updateEvolutionPipeline(pipeline) {
    const steps = document.querySelectorAll('#evo-pipeline .pipeline-step');
    if (!steps.length) return;
    if (pipeline.length > 0) {
        steps.forEach((el, i) => {
            const s = pipeline[i] || { status: 'pending' };
            el.className = `pipeline-step ${s.status}`;
        });
    }
}

function updateProposalTable(proposals) {
    const body = document.getElementById('evo-proposals-body');
    if (!body) return;
    if (proposals.length === 0) {
        body.innerHTML = '<tr><td colspan="4" class="muted-text">No proposals yet</td></tr>';
        return;
    }
    body.innerHTML = proposals.map(p => {
        const prioClass = p.priority === 'high' ? 'accent-text-red' : p.priority === 'low' ? 'accent-text-green' : 'accent-text-cyan';
        return `<tr><td>${escapeHtml(p.name)}</td><td class="${prioClass}">${capitalize(p.priority)}</td><td>${capitalize(p.status)}</td><td class="muted-text">${p.date || '--'}</td></tr>`;
    }).join('');
}

function updateEvolutionHistory(history) {
    const list = document.getElementById('evo-history-list');
    if (!list) return;
    if (history.length === 0) {
        list.innerHTML = '<div class="muted-text">No history yet</div>';
        return;
    }
    list.innerHTML = history.map(h => {
        const icon = h.success ? '<i class="fas fa-check-circle" style="color:#00ff88"></i>' : '<i class="fas fa-times-circle" style="color:#ff4466"></i>';
        return `<div class="evo-history-item">${icon} <span>${escapeHtml(h.event)}</span><span class="muted-text">${h.date || ''}</span></div>`;
    }).join('');
}

function updateCodeHealth(ch) {
    setSVGGauge('evo-test-ring', ch.test_pass_rate || 0);
    setSVGGauge('evo-lint-ring', ch.lint_score || 0);
    setSVGGauge('evo-complex-ring', ch.complexity || 0);
    setText('evo-test-val', `${ch.test_pass_rate || 0}%`);
    setText('evo-lint-val', ch.lint_score || 0);
    setText('evo-complex-val', ch.complexity || 0);
}

// ══════════════════════════════════════════════
// KNOWLEDGE PANEL HELPERS
// ══════════════════════════════════════════════
function updateCuriosityQueue(topics) {
    const container = document.getElementById('know-curiosity-queue');
    if (!container) return;
    if (topics.length === 0) {
        container.innerHTML = '<div class="muted-text">No curiosity topics yet</div>';
        return;
    }
    container.innerHTML = topics.map(t => {
        const urgency = Math.round((t.urgency || 0.5) * 100);
        const color = urgency > 70 ? '#ff4466' : urgency > 40 ? '#fbbf24' : '#00ff88';
        return `<div class="curiosity-item">
            <div class="curiosity-topic">${escapeHtml(t.topic)}</div>
            <div class="curiosity-meta"><span class="curiosity-source">${t.source || 'auto'}</span></div>
            <div class="urgency-bar"><div class="urgency-fill" style="width:${urgency}%;background:${color}"></div><span class="urgency-val">${urgency}%</span></div>
        </div>`;
    }).join('');
}

function updateRecentLearnings(learnings) {
    const container = document.getElementById('know-recent-learnings');
    if (!container) return;
    if (learnings.length === 0) {
        container.innerHTML = '<div class="muted-text">No recent learnings</div>';
        return;
    }
    container.innerHTML = learnings.map(l => {
        return `<div class="learning-item">
            <div class="learning-topic"><i class="fas fa-book-open"></i> ${escapeHtml(l.topic)}</div>
            <div class="learning-summary">${escapeHtml(l.summary || '')}</div>
            <div class="learning-date muted-text">${l.date || ''}</div>
        </div>`;
    }).join('');
}

function updateTopTopics(topics) {
    const container = document.getElementById('know-top-topics');
    if (!container) return;
    const entries = Object.entries(topics);
    if (entries.length === 0) {
        container.innerHTML = '<span class="muted-text">No topics yet</span>';
        return;
    }
    const maxCount = Math.max(...entries.map(([_, v]) => v), 1);
    container.innerHTML = entries.map(([name, count]) => {
        const size = 0.75 + (count / maxCount) * 0.8;
        const opacity = 0.5 + (count / maxCount) * 0.5;
        return `<span class="topic-tag" style="font-size:${size}rem;opacity:${opacity}">${escapeHtml(name)} <sup>${count}</sup></span>`;
    }).join(' ');
}

async function searchKnowledge() {
    const input = document.getElementById('know-search-input');
    const detail = document.getElementById('know-detail');
    if (!input || !detail) return;
    const q = input.value.trim();
    if (!q) { detail.innerHTML = ''; return; }
    detail.innerHTML = '<div class="muted-text"><i class="fas fa-spinner fa-spin"></i> Searching...</div>';
    try {
        const res = await fetch(`/api/knowledge/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        const results = data.results || [];
        if (results.length === 0) {
            detail.innerHTML = '<div class="muted-text">No results found</div>';
            return;
        }
        detail.innerHTML = results.map(r => {
            return `<div class="knowledge-result">
                <div class="result-topic"><i class="fas fa-file-alt"></i> ${escapeHtml(r.topic || r.title || '?')}</div>
                <div class="result-summary">${escapeHtml(r.summary || r.content || '')}</div>
            </div>`;
        }).join('');
    } catch (e) {
        detail.innerHTML = '<div class="muted-text">Search failed</div>';
    }
}

// ══════════════════════════════════════════════
// SYSTEM PANEL HELPERS
// ══════════════════════════════════════════════
function updateCoreBars(cores) {
    const container = document.getElementById('sys-core-bars');
    if (!container) return;
    if (cores.length === 0) {
        container.innerHTML = '<div class="muted-text">No core data</div>';
        return;
    }
    container.innerHTML = cores.map((pct, i) => {
        const color = pct > 80 ? '#ff4466' : pct > 50 ? '#fbbf24' : '#00d4ff';
        return `<div class="core-bar-row">
            <span class="core-label">C${i}</span>
            <div class="core-bar-track"><div class="core-bar-fill" style="width:${pct}%;background:${color}"></div></div>
            <span class="core-val">${Math.round(pct)}%</span>
        </div>`;
    }).join('');
}

function updateMemBreakdown(mb) {
    if (!mb.total_gb) return;
    const totalGb = mb.total_gb || 1;
    const usedPct = ((mb.used_gb || 0) / totalGb * 100).toFixed(1);
    const cachedPct = ((mb.cached_gb || 0) / totalGb * 100).toFixed(1);
    const freePct = (100 - usedPct - cachedPct).toFixed(1);
    const usedEl = document.getElementById('sys-mem-used');
    const cachedEl = document.getElementById('sys-mem-cached');
    const freeEl = document.getElementById('sys-mem-free');
    if (usedEl) usedEl.style.width = `${usedPct}%`;
    if (cachedEl) cachedEl.style.width = `${cachedPct}%`;
    if (freeEl) freeEl.style.width = `${freePct}%`;
    setText('sys-mem-used-val', mb.used_gb || 0);
    setText('sys-mem-cached-val', mb.cached_gb || 0);
    setText('sys-mem-free-val', mb.available_gb || 0);
}

function updateProcessTable(procs) {
    const body = document.getElementById('sys-proc-body');
    if (!body) return;
    if (procs.length === 0) {
        body.innerHTML = '<tr><td colspan="4" class="muted-text">No process data</td></tr>';
        return;
    }
    body.innerHTML = procs.map(p => {
        return `<tr><td>${escapeHtml(p.name || '?')}</td><td>${p.pid || '--'}</td><td>${(p.cpu_percent || 0).toFixed(1)}</td><td>${(p.memory_percent || 0).toFixed(1)}</td></tr>`;
    }).join('');
}

function updateBrainResources(br) {
    setText('sys-brain-mem', `${br.memory_mb || '--'} MB`);
    setText('sys-brain-threads', br.threads || '--');
    setText('sys-brain-cpu', `${br.cpu_pct || '--'}%`);
}

// ══════════════════════════════════════════════
// ANIMATED VALUE COUNTER
// ══════════════════════════════════════════════
function animateValue(elementId, newValue) {
    const el = document.getElementById(elementId);
    if (!el) return;
    newValue = parseInt(newValue) || 0;
    const oldValue = animatedValues[elementId] || 0;
    if (oldValue === newValue) { el.textContent = newValue.toLocaleString(); return; }
    animatedValues[elementId] = newValue;
    const diff = newValue - oldValue;
    const duration = Math.min(800, Math.max(200, Math.abs(diff) * 10));
    const startTime = performance.now();
    const animate = (now) => {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = Math.round(oldValue + diff * eased);
        el.textContent = current.toLocaleString();
        if (progress < 1) requestAnimationFrame(animate);
        else {
            el.textContent = newValue.toLocaleString();
            el.classList.add('value-flash');
            setTimeout(() => el.classList.remove('value-flash'), 600);
        }
    };
    requestAnimationFrame(animate);
}

// ══════════════════════════════════════════════
// PARTICLE BACKGROUND (Neural Network)
// ══════════════════════════════════════════════
function initParticleBackground() {
    particleCanvas = document.getElementById('particle-bg');
    if (!particleCanvas) return;
    particleCtx = particleCanvas.getContext('2d');
    resizeParticleCanvas();
    window.addEventListener('resize', resizeParticleCanvas);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
            x: Math.random() * particleCanvas.width,
            y: Math.random() * particleCanvas.height,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            r: Math.random() * 2 + 1,
            hue: Math.random() * 60 + 180, // cyan to blue range
        });
    }
    animateParticles();
}

function resizeParticleCanvas() {
    if (!particleCanvas) return;
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
}

function animateParticles() {
    if (!particleCtx || !particleCanvas) return;
    particleCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
    const w = particleCanvas.width, h = particleCanvas.height;
    // Update and draw particles
    for (const p of particles) {
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        particleCtx.beginPath();
        particleCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        particleCtx.fillStyle = `hsla(${p.hue}, 100%, 70%, 0.6)`;
        particleCtx.fill();
    }
    // Draw connections
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < CONNECTION_DISTANCE) {
                const alpha = (1 - dist / CONNECTION_DISTANCE) * 0.15;
                particleCtx.beginPath();
                particleCtx.moveTo(particles[i].x, particles[i].y);
                particleCtx.lineTo(particles[j].x, particles[j].y);
                particleCtx.strokeStyle = `rgba(0, 212, 255, ${alpha})`;
                particleCtx.lineWidth = 0.5;
                particleCtx.stroke();
            }
        }
    }
    particleRAF = requestAnimationFrame(animateParticles);
}

// ══════════════════════════════════════════════
// COMMAND PALETTE (Ctrl+K)
// ══════════════════════════════════════════════
const commandPaletteItems = [
    { label: 'Dashboard', icon: 'fa-th-large', action: () => switchPage('dashboard') },
    { label: 'Chat', icon: 'fa-comments', action: () => switchPage('chat') },
    { label: 'Mind', icon: 'fa-brain', action: () => switchPage('mind') },
    { label: 'Evolution', icon: 'fa-dna', action: () => switchPage('evolution') },
    { label: 'Knowledge', icon: 'fa-book', action: () => switchPage('knowledge') },
    { label: 'System', icon: 'fa-server', action: () => switchPage('system') },
    { label: 'Refresh Data', icon: 'fa-sync-alt', action: () => fetchStats() },
    { label: 'Export JSON', icon: 'fa-download', action: () => exportDashboardData() },
    { label: 'Toggle Fullscreen', icon: 'fa-expand', action: () => toggleFullscreen() },
    { label: 'Clear Chat', icon: 'fa-trash', action: () => { if (confirm('Clear chat history?')) clearChat(); } },
    { label: 'Logout', icon: 'fa-sign-out-alt', action: () => doLogout() },
];

function toggleCommandPalette() {
    const modal = document.getElementById('command-palette-modal');
    if (!modal) return;
    const isOpen = modal.classList.contains('active');
    if (isOpen) {
        modal.classList.remove('active');
    } else {
        modal.classList.add('active');
        const input = document.getElementById('cmd-palette-input');
        if (input) { input.value = ''; input.focus(); }
        renderCommandPaletteResults('');
    }
}

function renderCommandPaletteResults(query) {
    const list = document.getElementById('cmd-palette-results');
    if (!list) return;
    const q = query.toLowerCase().trim();
    const filtered = q ? commandPaletteItems.filter(i => i.label.toLowerCase().includes(q)) : commandPaletteItems;
    list.innerHTML = filtered.map((item, idx) => `
        <div class="cmd-palette-item${idx === 0 ? ' selected' : ''}" data-idx="${idx}" onclick="executeCommandPaletteItem(${commandPaletteItems.indexOf(item)})">
            <i class="fas ${item.icon}"></i>
            <span>${item.label}</span>
        </div>
    `).join('') || '<div class="cmd-palette-empty">No results found</div>';
}

function executeCommandPaletteItem(idx) {
    const item = commandPaletteItems[idx];
    if (!item) return;
    toggleCommandPalette();
    item.action();
    showToast(`Executed: ${item.label}`, 'info');
}

function handleCommandPaletteKey(e) {
    const list = document.getElementById('cmd-palette-results');
    if (!list) return;
    const items = list.querySelectorAll('.cmd-palette-item');
    let selectedIdx = -1;
    items.forEach((item, i) => { if (item.classList.contains('selected')) selectedIdx = i; });

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        const next = Math.min(selectedIdx + 1, items.length - 1);
        items.forEach(i => i.classList.remove('selected'));
        if (items[next]) items[next].classList.add('selected');
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        const prev = Math.max(selectedIdx - 1, 0);
        items.forEach(i => i.classList.remove('selected'));
        if (items[prev]) items[prev].classList.add('selected');
    } else if (e.key === 'Enter') {
        e.preventDefault();
        const selected = list.querySelector('.cmd-palette-item.selected');
        if (selected) selected.click();
    } else if (e.key === 'Escape') {
        toggleCommandPalette();
    }
}

// ══════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════════
let toastCounter = 0;
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const id = `toast-${toastCounter++}`;
    const icons = { info: 'fa-info-circle', success: 'fa-check-circle', warning: 'fa-exclamation-triangle', error: 'fa-times-circle' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.id = id;
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('show'));
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

// ══════════════════════════════════════════════
// KEYBOARD SHORTCUTS
// ══════════════════════════════════════════════
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ignore if typing in input/textarea
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            if (e.key === 'Escape') {
                e.target.blur();
                const modal = document.getElementById('command-palette-modal');
                if (modal && modal.classList.contains('active')) toggleCommandPalette();
            }
            return;
        }
        // Ctrl+K / Cmd+K — Command Palette
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            toggleCommandPalette();
            return;
        }
        // Number keys 1-6 for page navigation
        if (e.key >= '1' && e.key <= '6' && !e.ctrlKey && !e.altKey && !e.metaKey) {
            const pages = ['dashboard', 'chat', 'mind', 'evolution', 'knowledge', 'system'];
            switchPage(pages[parseInt(e.key) - 1]);
            return;
        }
        // R — Refresh
        if (e.key === 'r' || e.key === 'R') { fetchStats(); return; }
        // F — Fullscreen
        if (e.key === 'f' || e.key === 'F') { toggleFullscreen(); return; }
        // E — Export
        if (e.key === 'e' || e.key === 'E') { exportDashboardData(); return; }
        // ? — Show shortcuts help
        if (e.key === '?') { showToast('Shortcuts: 1-6=Pages, R=Refresh, F=Fullscreen, E=Export, Ctrl+K=Command', 'info', 5000); }
    });
}

// ══════════════════════════════════════════════
// EXPORT DASHBOARD DATA
// ══════════════════════════════════════════════
function exportDashboardData() {
    fetchWithAuth('/api/stats')
        .then(r => r.json())
        .then(data => {
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `nexus-dashboard-${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('Dashboard data exported!', 'success');
        })
        .catch(err => showToast(`Export failed: ${err.message}`, 'error'));
}

// ══════════════════════════════════════════════
// FULLSCREEN MODE
// ══════════════════════════════════════════════
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().then(() => {
            showToast('Entered fullscreen — press F or Esc to exit', 'info');
        }).catch(() => { });
    } else {
        document.exitFullscreen();
    }
}

// ══════════════════════════════════════════════
// ABILITIES PAGE
// ══════════════════════════════════════════════
let abilitiesData = [];
let abilitiesStats = {};

async function fetchAbilities() {
    try {
        // Fetch abilities list
        const abilitiesRes = await fetch('/api/abilities', { headers: getAuthHeaders() });
        const abilitiesJson = await abilitiesRes.json();
        abilitiesData = abilitiesJson.abilities || [];

        // Fetch ability stats
        const statsRes = await fetch('/api/abilities/stats', { headers: getAuthHeaders() });
        const statsJson = await statsRes.json();
        abilitiesStats = statsJson || {};

        // Fetch invocation history
        const historyRes = await fetch('/api/abilities/history?limit=20', { headers: getAuthHeaders() });
        const historyJson = await historyRes.json();

        updateAbilitiesUI(abilitiesData, abilitiesStats);
        updateAbilitiesHistory(historyJson.history || []);
        updateAbilityCategories(abilitiesData);

        setText('abilities-last-update', `Last update: ${new Date().toLocaleTimeString()}`);
    } catch (e) {
        console.error('Failed to fetch abilities:', e);
        showToast('Failed to load abilities', 'error');
    }
}

function updateAbilitiesUI(abilities, stats) {
    // Update stat cards
    const total = abilities.length;
    const successCount = stats.successful_invocations || 0;
    const cooldownCount = abilities.filter(a => isOnCooldown(a)).length;
    const totalInvokes = stats.total_invocations || 0;

    setText('abilities-total-count', total);
    setText('abilities-success-count', successCount);
    setText('abilities-cooldown-count', cooldownCount);
    setText('abilities-invoke-count', totalInvokes);

    // Update abilities grid
    const grid = document.getElementById('abilities-grid');
    if (!grid) return;

    if (abilities.length === 0) {
        grid.innerHTML = '<div class="muted-text">No abilities registered</div>';
        return;
    }

    grid.innerHTML = abilities.map((ability, idx) => {
        const catClass = getCategoryClass(ability.category);
        const riskClass = `risk-${ability.risk?.toLowerCase() || 'low'}`;
        const onCooldown = isOnCooldown(ability);
        const cooldownText = onCooldown ? getCooldownText(ability) : '';
        const catIcon = getCategoryIcon(ability.category);

        return `
            <div class="ability-card ${onCooldown ? 'on-cooldown' : ''} ability-animate-in" data-name="${escapeHtml(ability.name)}" style="animation-delay:${idx * 40}ms">
                <div class="ability-card-header">
                    <span class="ability-name">${catIcon} ${escapeHtml(ability.name)}</span>
                    <span class="ability-category-badge ${catClass}">${capitalize((ability.category || 'system').replace(/_/g, ' '))}</span>
                </div>
                <div class="ability-description">${escapeHtml(ability.description || 'No description')}</div>
                <div class="ability-meta">
                    <span class="ability-meta-item"><i class="fas fa-clock"></i> ${ability.cooldown_seconds || 0}s CD</span>
                    <span class="ability-meta-item"><i class="fas fa-bolt"></i> ${ability.invoke_count || 0} uses</span>
                    <span class="ability-risk ${riskClass}">${capitalize(ability.risk || 'low')}</span>
                </div>
                ${onCooldown ? `<div class="muted-text" style="font-size:.7rem;margin-top:6px;"><i class="fas fa-hourglass-half"></i> ${cooldownText}</div>` : ''}
                <button class="ability-invoke-btn" onclick="showInvokeModal('${escapeHtml(ability.name)}')" ${onCooldown ? 'disabled' : ''}>
                    <i class="fas fa-play"></i> Invoke
                </button>
            </div>
        `;
    }).join('');
}

function getCategoryClass(category) {
    const cat = (category || 'system').toLowerCase();
    const map = {
        'system': 'cat-system', 'system_control': 'cat-system',
        'cognition': 'cat-cognition', 'cognitive': 'cat-cognition',
        'communication': 'cat-communication', 'interaction': 'cat-communication',
        'self_evolution': 'cat-evolution', 'self_modification': 'cat-evolution',
        'learning': 'cat-learning', 'research': 'cat-research',
        'memory': 'cat-memory',
        'body': 'cat-body',
        'personality': 'cat-personality',
        'consciousness': 'cat-consciousness',
        'emotion': 'cat-emotion',
        'monitoring': 'cat-monitoring',
    };
    return map[cat] || 'cat-system';
}

function getCategoryIcon(category) {
    const cat = (category || 'system').toLowerCase();
    const icons = {
        'system': '⚙️', 'self_evolution': '🧬', 'learning': '📚', 'research': '🔬',
        'cognition': '🧠', 'memory': '💾', 'body': '🖥️', 'personality': '🎭',
        'consciousness': '✨', 'emotion': '💫', 'communication': '💬', 'monitoring': '📡',
    };
    return icons[cat] || '⚡';
}

function isOnCooldown(ability) {
    if (!ability.last_invoked) return false;
    const lastInvoked = new Date(ability.last_invoked);
    const cooldownMs = (ability.cooldown_seconds || 0) * 1000;
    return (Date.now() - lastInvoked.getTime()) < cooldownMs;
}

function getCooldownText(ability) {
    if (!ability.last_invoked) return '';
    const lastInvoked = new Date(ability.last_invoked);
    const cooldownMs = (ability.cooldown_seconds || 0) * 1000;
    const elapsed = Date.now() - lastInvoked.getTime();
    const remaining = Math.max(0, cooldownMs - elapsed);
    const seconds = Math.ceil(remaining / 1000);
    return `Cooldown: ${seconds}s remaining`;
}

function updateAbilitiesHistory(history) {
    const body = document.getElementById('abilities-history-body');
    if (!body) return;

    if (history.length === 0) {
        body.innerHTML = '<tr><td colspan="4" class="muted-text">No invocations yet</td></tr>';
        return;
    }

    body.innerHTML = history.map(h => {
        const successIcon = h.success
            ? '<i class="fas fa-check-circle" style="color:#00ff88"></i>'
            : '<i class="fas fa-times-circle" style="color:#ff4466"></i>';
        const time = h.timestamp ? new Date(h.timestamp).toLocaleTimeString() : '--';
        const duration = h.duration_ms ? `${h.duration_ms}ms` : '--';

        return `<tr>
            <td>${escapeHtml(h.ability_name || h.name || '?')}</td>
            <td class="muted-text">${time}</td>
            <td>${successIcon} ${h.success ? 'Success' : 'Failed'}</td>
            <td class="muted-text">${duration}</td>
        </tr>`;
    }).join('');
}

function updateAbilityCategories(abilities) {
    // Group abilities by category dynamically
    const grouped = {};
    abilities.forEach(a => {
        const cat = (a.category || 'system').toLowerCase();
        if (!grouped[cat]) grouped[cat] = [];
        grouped[cat].push(a);
    });

    // Update the 3 legacy static category boxes
    const legacyMap = {
        'abilities-cat-system': a => ['system', 'monitoring'].includes((a.category || '').toLowerCase()),
        'abilities-cat-cognitive': a => ['cognition', 'consciousness', 'emotion'].includes((a.category || '').toLowerCase()),
        'abilities-cat-interaction': a => ['communication', 'personality'].includes((a.category || '').toLowerCase()),
    };
    Object.entries(legacyMap).forEach(([elId, filterFn]) => {
        const el = document.getElementById(elId);
        if (!el) return;
        const filtered = abilities.filter(filterFn);
        if (filtered.length === 0) {
            el.innerHTML = '<div class="muted-text">No abilities</div>';
            return;
        }
        el.innerHTML = filtered.map(a => `
            <div class="ability-list-item">
                <span>${getCategoryIcon(a.category)} ${escapeHtml(a.name)}</span>
                <span class="ability-count">${a.invoke_count || 0}</span>
            </div>
        `).join('');
    });
}

// Invoke Modal
let currentInvokeAbility = null;

function showInvokeModal(abilityName) {
    const ability = abilitiesData.find(a => a.name === abilityName);
    if (!ability) return;

    currentInvokeAbility = ability;

    // Create modal overlay
    let overlay = document.getElementById('invoke-modal-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'invoke-modal-overlay';
        overlay.className = 'invoke-modal-overlay';
        overlay.onclick = (e) => { if (e.target === overlay) hideInvokeModal(); };
        document.body.appendChild(overlay);
    }

    // Build params HTML
    const params = ability.parameters || {};
    const paramsHtml = Object.keys(params).length > 0
        ? Object.entries(params).map(([key, spec]) => `
            <div class="invoke-param">
                <label>${escapeHtml(key)} ${spec.required ? '*' : ''}</label>
                <input type="text" id="invoke-param-${escapeHtml(key)}" placeholder="${escapeHtml(spec.default || spec.type || '')}">
            </div>
        `).join('')
        : '<div class="muted-text">No parameters required</div>';

    overlay.innerHTML = `
        <div class="invoke-modal">
            <h3><i class="fas fa-magic"></i> Invoke: ${escapeHtml(ability.name)}</h3>
            <div class="ability-desc">${escapeHtml(ability.description || 'No description')}</div>
            <div class="invoke-params">${paramsHtml}</div>
            <div id="invoke-result"></div>
            <div class="invoke-modal-actions">
                <button class="invoke-btn-cancel" onclick="hideInvokeModal()">Cancel</button>
                <button class="invoke-btn-confirm" onclick="executeInvoke()"><i class="fas fa-play"></i> Invoke</button>
            </div>
        </div>
    `;

    overlay.classList.add('active');
}

function hideInvokeModal() {
    const overlay = document.getElementById('invoke-modal-overlay');
    if (overlay) overlay.classList.remove('active');
    currentInvokeAbility = null;
}

async function executeInvoke() {
    if (!currentInvokeAbility) return;

    const resultEl = document.getElementById('invoke-result');
    if (resultEl) resultEl.innerHTML = '<div class="muted-text"><i class="fas fa-spinner fa-spin"></i> Invoking...</div>';

    // Collect params
    const params = {};
    const paramSpecs = currentInvokeAbility.parameters || {};
    Object.keys(paramSpecs).forEach(key => {
        const input = document.getElementById(`invoke-param-${key}`);
        if (input && input.value.trim()) {
            params[key] = input.value.trim();
        }
    });

    try {
        const res = await fetch('/api/abilities/invoke', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({ name: currentInvokeAbility.name, params })
        });
        const data = await res.json();

        if (data.success) {
            if (resultEl) {
                resultEl.innerHTML = `<div class="invoke-result success"><i class="fas fa-check-circle"></i> Success! ${escapeHtml(data.message || JSON.stringify(data.result || {}))}</div>`;
            }
            showToast(`Ability "${currentInvokeAbility.name}" executed successfully!`, 'success');

            // Refresh abilities after short delay
            setTimeout(fetchAbilities, 1000);
        } else {
            if (resultEl) {
                resultEl.innerHTML = `<div class="invoke-result error"><i class="fas fa-times-circle"></i> Failed: ${escapeHtml(data.error || 'Unknown error')}</div>`;
            }
            showToast(`Ability failed: ${data.error || 'Unknown error'}`, 'error');
        }
    } catch (e) {
        if (resultEl) {
            resultEl.innerHTML = `<div class="invoke-result error"><i class="fas fa-times-circle"></i> Connection error: ${escapeHtml(e.message)}</div>`;
        }
        showToast('Connection error', 'error');
    }
}

// Add Abilities to command palette
commandPaletteItems.push(
    { label: 'Abilities', icon: 'fa-magic', action: () => switchPage('abilities') },
    { label: 'Refresh Abilities', icon: 'fa-sync-alt', action: () => fetchAbilities() }
);

// ══════════════════════════════════════════════
// SUBSYSTEM PANELS — Fetch & Render
// ══════════════════════════════════════════════

let subsystemPollTimer = null;
let subsystemsLoaded = false;

function startSubsystemPolling() {
    if (subsystemPollTimer) return;
    fetchSubsystems();
    subsystemPollTimer = setInterval(fetchSubsystems, 5000); // Every 5s
}

async function fetchSubsystems() {
    const endpoints = [
        { url: '/api/immune', handler: updateImmunePanel },
        { url: '/api/autonomy', handler: updateAutonomyPanel },
        { url: '/api/worldmodel', handler: updateWorldModelPanel },
        { url: '/api/globalworkspace', handler: updateGlobalWorkspacePanel },
        { url: '/api/cognitiverouter', handler: updateCognitiveRouterPanel },
    ];

    const fetches = endpoints.map(async (ep) => {
        try {
            const res = await fetch(ep.url, { headers: getAuthHeaders() });
            if (res.ok) {
                const data = await res.json();
                ep.handler(data);
            }
        } catch (e) {
            // Silently fail — subsystem may not be loaded
        }
    });

    await Promise.all(fetches);
    subsystemsLoaded = true;
}

// ── Immune System Panel ──
function updateImmunePanel(data) {
    const running = data.running !== false && !data.error;
    const badge = document.getElementById('immune-status-badge');
    if (badge) {
        badge.textContent = running ? '🟢 Active' : '🔴 Not Loaded';
        badge.className = `subsystem-status-badge ${running ? 'status-active' : 'status-inactive'}`;
    }
    if (data.error) return;

    setText('immune-threats-blocked', data.threats_blocked || data.total_threats || 0);
    setText('immune-active-defenses', data.active_defenses || data.defense_count || 0);
    setText('immune-scans', data.scans_completed || data.total_scans || 0);
    setText('immune-last-scan', data.last_scan ? new Date(data.last_scan).toLocaleTimeString() : '--');
    const secLevel = data.security_level || data.threat_level || 'Normal';
    const secEl = document.getElementById('immune-security-level');
    if (secEl) {
        secEl.textContent = capitalize(secLevel);
        secEl.className = 'kv-val ' + (secLevel === 'critical' ? 'accent-text-red' : secLevel === 'elevated' ? 'accent-text-orange' : 'accent-text-green');
    }

    // Threat log
    const logEl = document.getElementById('immune-threat-log');
    if (logEl) {
        const threats = data.recent_threats || data.threat_log || [];
        if (threats.length > 0) {
            logEl.innerHTML = '<div class="subsystem-list-header"><i class="fas fa-exclamation-triangle"></i> Recent Threats</div>' +
                threats.slice(0, 5).map(t => {
                    const msg = typeof t === 'object' ? (t.description || t.type || JSON.stringify(t)) : String(t);
                    return `<div class="subsystem-list-item threat-item"><i class="fas fa-bug"></i> ${escapeHtml(msg)}</div>`;
                }).join('');
        } else {
            logEl.innerHTML = '<div class="subsystem-list-empty"><i class="fas fa-check-circle"></i> No threats detected</div>';
        }
    }
}

// ── Autonomy Engine Panel ──
function updateAutonomyPanel(data) {
    const running = data.running !== false && !data.error;
    const badge = document.getElementById('autonomy-status-badge');
    if (badge) {
        badge.textContent = running ? '🟢 Active' : '🔴 Not Loaded';
        badge.className = `subsystem-status-badge ${running ? 'status-active' : 'status-inactive'}`;
    }
    if (data.error) return;

    setText('autonomy-actions', data.autonomous_actions || data.actions_taken || data.total_actions || 0);
    setText('autonomy-current-goal', data.current_goal || data.active_goal || 'None');
    setText('autonomy-queue', data.decision_queue || data.queue_size || 0);
    setText('autonomy-decisions', data.decisions_made || data.total_decisions || 0);
    const level = data.autonomy_level || data.level || '--';
    setText('autonomy-level', typeof level === 'number' ? `${Math.round(level * 100)}%` : capitalize(String(level)));

    // Recent actions
    const actionsEl = document.getElementById('autonomy-recent-actions');
    if (actionsEl) {
        const actions = data.recent_actions || data.action_history || [];
        if (actions.length > 0) {
            actionsEl.innerHTML = '<div class="subsystem-list-header"><i class="fas fa-bolt"></i> Recent Actions</div>' +
                actions.slice(0, 5).map(a => {
                    const msg = typeof a === 'object' ? (a.description || a.action || a.type || JSON.stringify(a)) : String(a);
                    return `<div class="subsystem-list-item"><i class="fas fa-chevron-right"></i> ${escapeHtml(msg)}</div>`;
                }).join('');
        } else {
            actionsEl.innerHTML = '';
        }
    }
}

// ── World Model Panel ──
function updateWorldModelPanel(data) {
    const running = data.running !== false && !data.error;
    const badge = document.getElementById('worldmodel-status-badge');
    if (badge) {
        badge.textContent = running ? '🟢 Active' : '🔴 Not Loaded';
        badge.className = `subsystem-status-badge ${running ? 'status-active' : 'status-inactive'}`;
    }
    if (data.error) return;

    setText('wm-user-patterns', data.user_patterns || 0);
    setText('wm-emotional-patterns', data.emotional_patterns || 0);
    setText('wm-task-records', data.task_records || 0);
    setText('wm-predictions-made', data.predictions_made || 0);
    setText('wm-predictions-accurate', data.predictions_accurate || 0);

    // Environment state
    const env = data.environment || {};
    setText('wm-user-emotion', capitalize(env.user_emotional_state || '--'));
    const engagement = env.engagement_level;
    setText('wm-engagement', typeof engagement === 'number' ? `${Math.round(engagement * 100)}%` : '--');
    setText('wm-time-of-day', capitalize(env.time_of_day || '--'));
}

// ── Global Workspace Panel ──
function updateGlobalWorkspacePanel(data) {
    const running = data.running !== false && !data.error;
    const badge = document.getElementById('gw-status-badge');
    if (badge) {
        badge.textContent = running ? '🟢 Active' : '🔴 Not Loaded';
        badge.className = `subsystem-status-badge ${running ? 'status-active' : 'status-inactive'}`;
    }
    if (data.error) return;

    setText('gw-broadcasts', data.total_broadcasts || data.broadcasts || 0);
    setText('gw-coalitions', data.active_coalitions || data.coalitions || 0);
    setText('gw-winner', data.current_winner || data.winner || '--');
    setText('gw-focus', data.attention_focus || data.focus || '--');
    setText('gw-processes', data.registered_processes || data.processes || data.subscribers || 0);
    const integration = data.integration_level || data.integration;
    setText('gw-integration', typeof integration === 'number' ? `${Math.round(integration * 100)}%` : capitalize(String(integration || '--')));

    // Recent broadcasts
    const broadcastEl = document.getElementById('gw-recent-broadcasts');
    if (broadcastEl) {
        const broadcasts = data.recent_broadcasts || [];
        if (broadcasts.length > 0) {
            broadcastEl.innerHTML = '<div class="subsystem-list-header"><i class="fas fa-broadcast-tower"></i> Recent Broadcasts</div>' +
                broadcasts.slice(0, 5).map(b => {
                    const msg = typeof b === 'object' ? (b.content || b.type || JSON.stringify(b)) : String(b);
                    return `<div class="subsystem-list-item"><i class="fas fa-signal"></i> ${escapeHtml(msg)}</div>`;
                }).join('');
        } else {
            broadcastEl.innerHTML = '';
        }
    }
}

// ── Cognitive Router Panel ──
function updateCognitiveRouterPanel(data) {
    const routerData = data.router || data;
    const enginesData = data.engines || [];

    const running = routerData.running !== false && !data.error;
    const badge = document.getElementById('cr-status-badge');
    if (badge) {
        badge.textContent = running ? '🟢 Active' : '🔴 Not Loaded';
        badge.className = `subsystem-status-badge ${running ? 'status-active' : 'status-inactive'}`;
    }
    if (data.error) return;

    setText('cr-routes', routerData.routes_processed || routerData.total_routes || 0);
    setText('cr-active-engine', routerData.active_engine || routerData.last_engine || '--');
    const avgTime = routerData.avg_route_time || routerData.average_time;
    setText('cr-avg-time', typeof avgTime === 'number' ? `${avgTime.toFixed(1)}ms` : '--');
    const cacheRate = routerData.cache_hit_rate || routerData.cache_rate;
    setText('cr-cache-rate', typeof cacheRate === 'number' ? `${Math.round(cacheRate * 100)}%` : '--');

    // Engine count
    const engineCount = enginesData.length || routerData.engine_count || routerData.total_engines || 0;
    setText('cr-engine-count', `${engineCount} engines`);

    // Engine grid
    const gridEl = document.getElementById('cr-engine-grid');
    if (gridEl && enginesData.length > 0) {
        gridEl.innerHTML = enginesData.map(engine => {
            const name = typeof engine === 'object' ? (engine.name || engine.engine || 'Unknown') : String(engine);
            const status = typeof engine === 'object' ? (engine.status || 'ready') : 'ready';
            const invocations = typeof engine === 'object' ? (engine.invocations || engine.invoke_count || 0) : 0;
            const statusClass = status === 'active' ? 'engine-active' : status === 'error' ? 'engine-error' : 'engine-ready';
            const displayName = name.replace(/_/g, ' ').replace(/engine$/i, '').trim();
            return `<div class="engine-badge ${statusClass}" title="${escapeHtml(name)}: ${invocations} invocations">
                <span class="engine-badge-name">${escapeHtml(capitalize(displayName))}</span>
                <span class="engine-badge-count">${invocations}</span>
            </div>`;
        }).join('');
    } else if (gridEl && engineCount > 0) {
        gridEl.innerHTML = `<div class="muted-text">${engineCount} engines registered (details loading...)</div>`;
    }
}

// Wire subsystem polling into system page visibility
const _origSwitchPage = switchPage;
// Override switchPage to trigger subsystem fetch when System page is shown
// and abilities fetch when Abilities page is shown
(function () {
    const original = switchPage;
    switchPage = function (pageId) {
        original(pageId);
        if (pageId === 'system' && !subsystemsLoaded) {
            fetchSubsystems();
        }
        if (pageId === 'abilities') {
            fetchAbilities();
        }
        if (pageId === 'settings') {
            loadSettingsPage();
        }
    };
})();

// Also start subsystem polling alongside main polling
(function () {
    // Delay subsystem poll start to avoid overloading on page load
    setTimeout(startSubsystemPolling, 4000);
})();

// ══════════════════════════════════════════════
// SETTINGS PAGE
// ══════════════════════════════════════════════

async function loadSettingsPage() {
    try {
        const res = await fetch('/api/user/profile', { headers: getAuthHeaders() });
        if (!res.ok) return;
        const data = await res.json();
        const p = data.profile || {};

        // Profile card
        setText('settings-display-name', p.display_name || p.username || 'User');
        setText('settings-username', '@' + (p.username || 'user'));
        if (p.created_at) {
            setText('settings-member-since', 'Member since ' + new Date(p.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }));
        }

        // Avatar
        const avatarEl = document.getElementById('settings-avatar');
        if (avatarEl) {
            if (p.profile_picture) {
                avatarEl.innerHTML = `<img src="${p.profile_picture}" alt="Avatar">`;
            } else {
                avatarEl.innerHTML = '<i class="fas fa-user"></i>';
            }
        }

        // Edit form
        const nameInput = document.getElementById('settings-edit-displayname');
        if (nameInput) nameInput.value = p.display_name || '';
        const bioInput = document.getElementById('settings-edit-bio');
        if (bioInput) bioInput.value = p.bio || '';

        // Account info
        setText('settings-acct-username', p.username || '—');
        setText('settings-acct-display', p.display_name || '—');
        setText('settings-acct-created', p.created_at ? new Date(p.created_at).toLocaleDateString() : '—');
        setText('settings-acct-login', p.last_login ? new Date(p.last_login).toLocaleString() : '—');
    } catch (e) {
        console.warn('Failed to load settings:', e);
    }
}

async function saveProfile() {
    const displayName = document.getElementById('settings-edit-displayname')?.value?.trim();
    const bio = document.getElementById('settings-edit-bio')?.value?.trim() || '';
    const msgEl = document.getElementById('profile-save-msg');

    if (!displayName) {
        showSettingsMsg(msgEl, 'Display name is required', 'error');
        return;
    }

    try {
        const res = await fetch('/api/user/profile', {
            method: 'PUT',
            headers: getAuthHeaders(),
            body: JSON.stringify({ display_name: displayName, bio })
        });
        const data = await res.json();

        if (res.ok) {
            showSettingsMsg(msgEl, 'Profile saved successfully!', 'success');
            // Update header/sidebar
            currentUser = { ...currentUser, display_name: displayName };
            setText('header-username', displayName);
            setText('sidebar-username', displayName);
            setText('settings-display-name', displayName);
            setText('settings-acct-display', displayName);
        } else {
            showSettingsMsg(msgEl, data.error || 'Failed to save', 'error');
        }
    } catch (e) {
        showSettingsMsg(msgEl, 'Connection error', 'error');
    }
}

async function changePassword() {
    const currentPw = document.getElementById('settings-current-pw')?.value || '';
    const newPw = document.getElementById('settings-new-pw')?.value || '';
    const confirmPw = document.getElementById('settings-confirm-pw')?.value || '';
    const msgEl = document.getElementById('password-save-msg');

    if (!currentPw || !newPw) {
        showSettingsMsg(msgEl, 'Please fill in all password fields', 'error');
        return;
    }
    if (newPw !== confirmPw) {
        showSettingsMsg(msgEl, 'New passwords do not match', 'error');
        return;
    }
    if (newPw.length < 4) {
        showSettingsMsg(msgEl, 'Password must be at least 4 characters', 'error');
        return;
    }

    try {
        const res = await fetch('/api/user/password', {
            method: 'PUT',
            headers: getAuthHeaders(),
            body: JSON.stringify({ current_password: currentPw, new_password: newPw })
        });
        const data = await res.json();

        if (res.ok) {
            showSettingsMsg(msgEl, 'Password changed successfully!', 'success');
            // Clear fields
            document.getElementById('settings-current-pw').value = '';
            document.getElementById('settings-new-pw').value = '';
            document.getElementById('settings-confirm-pw').value = '';
        } else {
            showSettingsMsg(msgEl, data.error || 'Failed to change password', 'error');
        }
    } catch (e) {
        showSettingsMsg(msgEl, 'Connection error', 'error');
    }
}

async function uploadAvatar(event) {
    const file = event?.target?.files?.[0];
    if (!file) return;

    // Validate file type and size
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    if (file.size > 2 * 1024 * 1024) {
        alert('Image must be under 2MB');
        return;
    }

    const reader = new FileReader();
    reader.onload = async function (e) {
        const base64 = e.target.result; // data:image/...;base64,...

        // Show preview immediately
        const avatarEl = document.getElementById('settings-avatar');
        if (avatarEl) avatarEl.innerHTML = `<img src="${base64}" alt="Avatar">`;

        try {
            const res = await fetch('/api/user/avatar', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ avatar: base64 })
            });
            const data = await res.json();
            if (!res.ok) {
                alert(data.error || 'Failed to upload avatar');
            }
        } catch (err) {
            alert('Connection error while uploading avatar');
        }
    };
    reader.readAsDataURL(file);
}

function showSettingsMsg(el, text, type) {
    if (!el) return;
    el.textContent = text;
    el.className = 'settings-msg ' + (type === 'success' ? 'msg-success' : 'msg-error');
    setTimeout(() => {
        el.textContent = '';
        el.className = 'settings-msg';
    }, 4000);
}

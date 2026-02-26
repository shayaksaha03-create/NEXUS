/*
 * NEXUS AI Mobile — Connection & WebView Logic
 */

// ══════════════════════════════════════════════
// CONSTANTS & STATE
// ══════════════════════════════════════════════
const STORAGE_KEY = 'nexus_server_url';
const HISTORY_KEY = 'nexus_server_history';
const MAX_HISTORY = 5;
const HEALTH_TIMEOUT = 8000;  // ms

let currentServerUrl = '';
let isConnected = false;

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    loadRecentServers();

    // Auto-connect if we have a saved URL
    const savedUrl = localStorage.getItem(STORAGE_KEY);
    if (savedUrl) {
        document.getElementById('server-url').value = savedUrl;
        // Auto-connect after a brief delay
        setTimeout(() => connectToServer(), 500);
    }

    // Enter key to connect
    const urlInput = document.getElementById('server-url');
    urlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') connectToServer();
    });

    // Close fab menu when tapping outside
    document.addEventListener('click', (e) => {
        const fabMenu = document.getElementById('fab-menu');
        if (fabMenu && !fabMenu.contains(e.target)) {
            document.getElementById('fab-options').classList.remove('open');
        }
    });
});

// ══════════════════════════════════════════════
// CONNECTION LOGIC
// ══════════════════════════════════════════════
async function connectToServer() {
    const urlInput = document.getElementById('server-url');
    const connectBtn = document.getElementById('connect-btn');
    const statusEl = document.getElementById('connect-status');
    const errorEl = document.getElementById('connect-error');

    let url = (urlInput.value || '').trim();
    if (!url) {
        showError('Please enter a server URL');
        return;
    }

    // Add protocol if missing
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'http://' + url;
    }
    // Remove trailing slash
    url = url.replace(/\/+$/, '');

    // Validate URL format
    try {
        new URL(url);
    } catch (e) {
        showError('Invalid URL format');
        return;
    }

    // Update UI — show loading
    connectBtn.disabled = true;
    connectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Connecting...</span>';
    errorEl.textContent = '';
    statusEl.className = 'connect-status';
    statusEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Checking server...</span>';

    try {
        // Check if server is reachable via /api/health
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT);

        const response = await fetch(`${url}/api/health`, {
            signal: controller.signal,
            mode: 'cors',
        });
        clearTimeout(timeout);

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        if (data.status !== 'ok') {
            throw new Error('Server health check failed');
        }

        // Success — save and connect
        currentServerUrl = url;
        isConnected = true;

        // Save to history
        if (document.getElementById('remember-server').checked) {
            localStorage.setItem(STORAGE_KEY, url);
        }
        addToHistory(url);

        // Show loading overlay
        showLoading(url);

        // Load the NEXUS web UI in the iframe
        setTimeout(() => {
            loadWebUI(url);
        }, 800);

    } catch (e) {
        let errorMsg = 'Could not connect to server';
        if (e.name === 'AbortError') {
            errorMsg = 'Connection timed out — check the URL';
        } else if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
            errorMsg = 'Network error — is the server running?';
        } else if (e.message.includes('CORS')) {
            errorMsg = 'CORS error — server may not allow this connection';
        } else if (e.message) {
            errorMsg = e.message;
        }

        showError(errorMsg);
        statusEl.className = 'connect-status error';
        statusEl.innerHTML = `<i class="fas fa-exclamation-triangle"></i> <span>Connection failed</span>`;
    } finally {
        connectBtn.disabled = false;
        connectBtn.innerHTML = '<i class="fas fa-bolt"></i> <span>Connect</span>';
    }
}

function loadWebUI(url) {
    const frame = document.getElementById('nexus-frame');
    const connectScreen = document.getElementById('connect-screen');
    const loadingOverlay = document.getElementById('loading-overlay');
    const fabMenu = document.getElementById('fab-menu');

    frame.src = url;
    frame.style.display = 'block';

    frame.onload = () => {
        // Hide loading and connection screen
        loadingOverlay.style.display = 'none';
        connectScreen.style.display = 'none';
        fabMenu.style.display = 'block';
    };

    // Fallback — hide loading after timeout even if onload doesn't fire
    setTimeout(() => {
        loadingOverlay.style.display = 'none';
        connectScreen.style.display = 'none';
        fabMenu.style.display = 'block';
    }, 10000);
}

function showLoading(url) {
    const overlay = document.getElementById('loading-overlay');
    const urlLabel = document.getElementById('loading-url');
    overlay.style.display = 'flex';
    urlLabel.textContent = url;
}

function showError(msg) {
    const errorEl = document.getElementById('connect-error');
    errorEl.textContent = msg;
}

// ══════════════════════════════════════════════
// FAB MENU ACTIONS
// ══════════════════════════════════════════════
function toggleFabMenu() {
    const options = document.getElementById('fab-options');
    options.classList.toggle('open');
}

function refreshServer() {
    const frame = document.getElementById('nexus-frame');
    if (frame && currentServerUrl) {
        frame.src = currentServerUrl;
    }
    document.getElementById('fab-options').classList.remove('open');
}

function disconnectServer() {
    const frame = document.getElementById('nexus-frame');
    const connectScreen = document.getElementById('connect-screen');
    const fabMenu = document.getElementById('fab-menu');
    const statusEl = document.getElementById('connect-status');

    frame.src = '';
    frame.style.display = 'none';
    connectScreen.style.display = 'flex';
    fabMenu.style.display = 'none';
    isConnected = false;
    currentServerUrl = '';

    statusEl.className = 'connect-status';
    statusEl.innerHTML = '<i class="fas fa-link"></i> <span>Enter your NEXUS server address</span>';

    document.getElementById('fab-options').classList.remove('open');
}

function showServerInfo() {
    document.getElementById('fab-options').classList.remove('open');

    const overlay = document.createElement('div');
    overlay.className = 'toast-overlay';
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

    const connectedAt = new Date().toLocaleTimeString();

    overlay.innerHTML = `
        <div class="toast-box">
            <h3><i class="fas fa-server"></i> Server Info</h3>
            <div class="info-row">
                <span class="info-key">URL</span>
                <span class="info-val">${escapeHtml(currentServerUrl)}</span>
            </div>
            <div class="info-row">
                <span class="info-key">Status</span>
                <span class="info-val" style="color: var(--accent-green)">● Connected</span>
            </div>
            <div class="info-row">
                <span class="info-key">Session</span>
                <span class="info-val">${connectedAt}</span>
            </div>
            <button class="close-toast" onclick="this.closest('.toast-overlay').remove()">Close</button>
        </div>
    `;

    document.body.appendChild(overlay);
}

// ══════════════════════════════════════════════
// RECENT SERVERS
// ══════════════════════════════════════════════
function loadRecentServers() {
    const history = getHistory();
    const container = document.getElementById('recent-servers');
    const list = document.getElementById('recent-list');

    if (history.length === 0) {
        container.style.display = 'none';
        return;
    }

    container.style.display = 'block';
    list.innerHTML = history.map((url, i) => `
        <div class="recent-item" onclick="selectRecentServer('${escapeHtml(url)}')">
            <span class="recent-item-url">${escapeHtml(url)}</span>
            <button class="recent-item-remove" onclick="event.stopPropagation(); removeFromHistory(${i})">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `).join('');
}

function selectRecentServer(url) {
    document.getElementById('server-url').value = url;
    connectToServer();
}

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    } catch {
        return [];
    }
}

function addToHistory(url) {
    let history = getHistory();
    // Remove if already present
    history = history.filter(u => u !== url);
    // Add to front
    history.unshift(url);
    // Keep max
    if (history.length > MAX_HISTORY) history = history.slice(0, MAX_HISTORY);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    loadRecentServers();
}

function removeFromHistory(index) {
    let history = getHistory();
    history.splice(index, 1);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    loadRecentServers();
}

// ══════════════════════════════════════════════
// PARTICLE BACKGROUND (lightweight version)
// ══════════════════════════════════════════════
let particles = [];
let particleCtx = null;
let particleAnimFrame = null;

function initParticles() {
    const canvas = document.getElementById('particle-bg');
    if (!canvas) return;

    particleCtx = canvas.getContext('2d');
    resizeCanvas(canvas);
    window.addEventListener('resize', () => resizeCanvas(canvas));

    // Create particles
    const count = 40;
    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            size: Math.random() * 2 + 0.5,
            alpha: Math.random() * 0.4 + 0.1,
        });
    }

    animateParticles(canvas);
}

function resizeCanvas(canvas) {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function animateParticles(canvas) {
    const ctx = particleCtx;
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Update and draw particles
    for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around
        if (p.x < 0) p.x = w;
        if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h;
        if (p.y > h) p.y = 0;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 212, 255, ${p.alpha})`;
        ctx.fill();
    }

    // Draw connections
    const connDist = 100;
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < connDist) {
                const alpha = (1 - dist / connDist) * 0.12;
                ctx.beginPath();
                ctx.moveTo(particles[i].x, particles[i].y);
                ctx.lineTo(particles[j].x, particles[j].y);
                ctx.strokeStyle = `rgba(0, 212, 255, ${alpha})`;
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }
        }
    }

    particleAnimFrame = requestAnimationFrame(() => animateParticles(canvas));
}

// ══════════════════════════════════════════════
// UTILS
// ══════════════════════════════════════════════
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

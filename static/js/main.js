let sensorChartInstance = null;
let sensorChartFullInstance = null;
let motorsCache = [];
let logsCache = [];
let currentDevice = null;
let expandedParam = null;
let fullSparkCharts = {};

document.addEventListener("DOMContentLoaded", () => {
    initChart();
    loadDevices();
    fetchAndRenderAlerts();
    loadMotors();
    loadSensors();
    loadLogs();
});

const SENSOR_LABELS = ['T-30m', 'T-25m', 'T-20m', 'T-15m', 'T-10m', 'T-5m', 'Now'];
const SENSOR_TEMP = [45, 46, 45, 50, 58, 65, 72];
const SENSOR_VIB = [1.2, 1.2, 1.3, 1.5, 2.1, 2.8, 3.5];
const SENSOR_CURR = [22, 23, 22, 24, 26, 28, 29];

function buildSensorDatasets() {
    return [
        { label: 'Temperature (C)', data: SENSOR_TEMP, borderColor: '#f97316', backgroundColor: 'rgba(249, 115, 22, 0.1)', borderWidth: 2, tension: 0.4, fill: true },
        { label: 'Vibration (mm/s)', data: SENSOR_VIB, borderColor: '#a855f7', borderWidth: 2, tension: 0.4, fill: false },
        { label: 'Current (A)', data: SENSOR_CURR, borderColor: '#38bdf8', borderWidth: 2, tension: 0.4, fill: false }
    ];
}

function chartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#9ca3af', boxWidth: 12, padding: 16 } } },
        scales: {
            y: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' } },
            x: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' } }
        }
    };
}

function initChart() {
    const ctx = document.getElementById('sensorChart').getContext('2d');
    sensorChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: SENSOR_LABELS, datasets: buildSensorDatasets() },
        options: chartOptions()
    });
}

function initFullChart() {
    const canvas = document.getElementById('sensorChartFull');
    if (!canvas || sensorChartFullInstance) return;
    const ctx = canvas.getContext('2d');
    sensorChartFullInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: SENSOR_LABELS, datasets: buildSensorDatasets() },
        options: chartOptions()
    });
}

/* COLLAPSIBLE CHART SECTIONS */
function toggleChart(containerId, btnEl) {
    const container = document.getElementById(containerId);
    container.classList.toggle('collapsed');
    const icon = btnEl.querySelector('i');
    if (container.classList.contains('collapsed')) {
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
    } else {
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
    }
}

/* PARAM DETAIL TOGGLE */
function toggleParamDetail(paramKey, btnEl) {
    const panel = document.getElementById('detail-' + paramKey);
    const isOpen = panel.classList.contains('open');

    document.querySelectorAll('.param-detail-panel').forEach(p => p.classList.remove('open'));
    document.querySelectorAll('.param-expand-btn').forEach(b => b.classList.remove('open'));

    if (!isOpen) {
        panel.classList.add('open');
        btnEl.classList.add('open');
        expandedParam = paramKey;
        renderFullSpark(paramKey);
    } else {
        expandedParam = null;
    }
}

function renderFullSpark(paramKey) {
    const url = currentDevice ? `/api/history?device=${encodeURIComponent(currentDevice)}&points=30` : '/api/history?points=30';
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const colorMap = { temp: '#f97316', vib: '#a855f7', volt: '#38bdf8', rpm: '#22c55e' };
            const fieldMap = { temp: 'temperature', vib: 'vibration', volt: 'voltage', rpm: 'rpm' };
            const canvasId = 'spark-' + paramKey + '-full';

            if (fullSparkCharts[canvasId]) fullSparkCharts[canvasId].destroy();

            const ctx = document.getElementById(canvasId).getContext('2d');
            fullSparkCharts[canvasId] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [{
                        data: data[fieldMap[paramKey]],
                        borderColor: colorMap[paramKey],
                        backgroundColor: colorMap[paramKey] + '22',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' } },
                        y: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' } }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading full sparkline:', error));
}

/* DEVICE LIST */
function loadDevices() {
    fetch('/api/devices')
        .then(r => r.json())
        .then(data => {
            const select = document.getElementById('device-select');
            select.innerHTML = '';
            data.devices.forEach(deviceId => {
                const option = document.createElement('option');
                option.value = deviceId;
                option.innerText = `Induction Motor ${deviceId}`;
                select.appendChild(option);
            });

            currentDevice = data.devices[0];
            select.value = currentDevice;
            fetchAndUpdateMetrics();
        })
        .catch(error => console.error('Error loading devices:', error));
}

function fetchAndUpdateMetrics() {
    const url = currentDevice ? `/api/status?device=${encodeURIComponent(currentDevice)}` : '/api/status';

    fetch(url)
        .then(response => response.json())
        .then(data => {
            document.getElementById('val-health').innerText = data.health_score + "%";
            document.getElementById('val-rul').innerHTML = data.rul_hours.toLocaleString() + ' <span class="unit">Hrs</span>';
            document.getElementById('val-falsealarm').innerText = data.false_alarm_rate + "%";
            document.getElementById('val-failprob').innerText = data.failure_probability + "%";
            document.getElementById('val-failprob-window').innerText = data.risk_window_label || 'Based on current sensor pattern';
            document.getElementById('val-risklevel').innerText = data.failure_probability + "%";
            document.getElementById('val-temperature').innerHTML = data.temperature + ' <span class="unit">C</span>';
            document.getElementById('val-vibration').innerHTML = data.vibration + ' <span class="unit">mm/s</span>';
            document.getElementById('val-current').innerHTML = data.current + ' <span class="unit">A</span>';
            document.getElementById('val-pressure').innerHTML = data.pressure + ' <span class="unit">bar</span>';
            document.getElementById('val-recommendation').innerText = data.recommendation || 'System operating within normal parameters.';

            const healthNote = document.getElementById('val-health-note');
            if (data.health_score >= 70) {
                healthNote.innerText = 'Normal operating range';
                healthNote.className = 'card-delta positive';
            } else if (data.health_score >= 40) {
                healthNote.innerText = 'Reduced performance detected';
                healthNote.className = 'card-delta';
            } else {
                healthNote.innerText = 'Immediate attention required';
                healthNote.className = 'card-delta';
                healthNote.style.color = 'var(--danger)';
            }

            currentDevice = data.device;
            loadSparklines();
            
            // Re-render full spark if expanded
            if (expandedParam) renderFullSpark(expandedParam);
        })
        .catch(error => console.error('Error updating status:', error));
}

function fetchAndRenderAlerts() {
    const url = currentDevice ? `/api/alerts?device=${encodeURIComponent(currentDevice)}` : '/api/alerts';

    fetch(url)
        .then(response => response.json())
        .then(alerts => {
            const container = document.getElementById('alert-list');
            container.innerHTML = '';
            alerts.forEach(a => {
                const row = document.createElement('div');
                row.className = 'alert-row';
                row.innerHTML = `
                    <div class="alert-info">
                        <div class="icon-box ${a.type}"><i class="fa-solid ${a.icon}"></i></div>
                        <div><h4>${a.title}</h4><p>${a.description}</p></div>
                    </div>
                    <div class="alert-action">
                        <span class="time">${a.time}</span>
                        ${a.action ? `<button class="btn-action" onclick="investigateAlert('${a.device_id || ''}')">${a.action}</button>` : ''}
                    </div>`;
                container.appendChild(row);
            });
        })
        .catch(error => console.error('Error loading alerts:', error));
}

/* MOTOR ASSETS */
function loadMotors() {
    fetch('/api/motors')
        .then(r => r.json())
        .then(data => {
            motorsCache = data.motors;
            document.getElementById('sum-total').innerText = data.summary.total;
            document.getElementById('sum-active').innerText = data.summary.active;
            document.getElementById('sum-idle').innerText = data.summary.idle;
            document.getElementById('sum-maintenance').innerText = data.summary.maintenance;
            document.getElementById('sum-stopped').innerText = data.summary.stopped;
            renderMotorsTable();
        })
        .catch(error => console.error('Error loading motors:', error));
}

function healthColor(score) {
    if (score >= 80) return '#22c55e';
    if (score >= 60) return '#f97316';
    if (score >= 40) return '#f59e0b';
    return '#ef4444';
}

function renderMotorsTable() {
    const search = document.getElementById('asset-search').value.toLowerCase();
    const statusFilter = document.getElementById('asset-status-filter').value;
    const body = document.getElementById('motors-table-body');
    body.innerHTML = '';

    const filtered = motorsCache.filter(m => {
        const matchSearch = m.name.toLowerCase().includes(search) || m.id.toLowerCase().includes(search);
        const matchStatus = statusFilter === 'all' || m.status === statusFilter;
        return matchSearch && matchStatus;
    });

    if (filtered.length === 0) {
        body.innerHTML = '<tr><td colspan="6" style="text-align:center; color: var(--text-muted); padding: 24px;">No motors found.</td></tr>';
        return;
    }

    filtered.forEach(m => {
        const color = healthColor(m.health_score);
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>
                <div class="motor-cell">
                    <div class="motor-icon"><i class="fa-solid fa-microchip"></i></div>
                    <div>
                        <div class="motor-name">${m.name}</div>
                        <div class="motor-sub">${m.id} - ${m.location}</div>
                    </div>
                </div>
            </td>
            <td><span class="status-pill ${m.status}"><span class="dot"></span>${m.status}</span></td>
            <td>
                <div class="health-cell">
                    <div class="health-bar-track"><div class="health-bar-fill" style="width:${m.health_score}%; background:${color};"></div></div>
                    <span class="health-num" style="color:${color};">${m.health_score}/100</span>
                </div>
            </td>
            <td>${m.rul_hours.toLocaleString()}</td>
            <td>${m.last_update}</td>
            <td><button class="row-action-btn" onclick="investigateAlert('${m.id}')" title="View details"><i class="fa-solid fa-chevron-right"></i></button></td>
        `;
        body.appendChild(tr);
    });
}

/* SENSOR DATA */
function loadSensors() {
    fetch('/api/sensors')
        .then(r => r.json())
        .then(data => {
            const body = document.getElementById('sensors-table-body');
            body.innerHTML = '';
            data.forEach(s => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>
                        <div class="motor-cell">
                            <div class="motor-icon"><i class="fa-solid fa-microchip"></i></div>
                            <div class="motor-name">${s.motor_name}</div>
                        </div>
                    </td>
                    <td>${s.temperature}</td>
                    <td>${s.vibration}</td>
                    <td>${s.current}</td>
                    <td>${s.pressure}</td>
                    <td><span class="status-pill ${s.status}"><span class="dot"></span>${s.status}</span></td>
                `;
                body.appendChild(tr);
            });
            initFullChart();
        })
        .catch(error => console.error('Error loading sensors:', error));
}

/* AI ALERTS / LOGS */
function loadLogs() {
    fetch('/api/logs')
        .then(r => r.json())
        .then(data => {
            logsCache = data;
            renderLogsList();
        })
        .catch(error => console.error('Error loading logs:', error));
}

function renderLogsList() {
    const filter = document.getElementById('log-severity-filter').value;
    const container = document.getElementById('logs-list');
    container.innerHTML = '';

    const filtered = logsCache.filter(l => filter === 'all' || l.type === filter);

    if (filtered.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); font-size: 12px; padding: 12px 0;">No logs match this filter.</p>';
        return;
    }

    filtered.forEach(l => {
        const row = document.createElement('div');
        row.className = 'log-row';
        row.innerHTML = `
            <div class="log-info">
                <div class="icon-box ${l.type}"><i class="fa-solid ${l.icon}"></i></div>
                <div>
                    <h4>${l.title}</h4>
                    <p>${l.description}</p>
                    <div class="log-meta">${l.device} - ${l.time}</div>
                </div>
            </div>
            <span class="severity-tag ${l.type}">${l.type}</span>
        `;
        container.appendChild(row);
    });
}

/* TAB SWITCHING */
function switchTab(tabName, navEl) {
    const items = document.querySelectorAll('.nav-item');
    items.forEach(item => item.classList.remove('active'));
    if (navEl) {
        navEl.classList.add('active');
    } else {
        const target = document.querySelector(`.nav-item[data-tab="${tabName}"]`);
        if (target) target.classList.add('active');
    }

    const pages = ['dashboard', 'assets', 'sensors', 'alerts', 'settings'];
    pages.forEach(page => {
        const el = document.getElementById('page-' + page);
        if (el) el.style.display = (page === tabName) ? 'flex' : 'none';
    });

    if (tabName === 'sensors') initFullChart();
}

function selectDevice(deviceId) {
    currentDevice = deviceId;
    fetchAndUpdateMetrics();
    fetchAndRenderAlerts();
}

function refreshData() {
    fetchAndUpdateMetrics();
    fetchAndRenderAlerts();
}

function triggerBell() { alert("Placeholder notification action triggered."); }

function investigateAlert(deviceId) {
    if (deviceId) {
        selectDevice(deviceId);
        const select = document.getElementById('device-select');
        select.value = deviceId;
        switchTab('sensors');
    } else {
        switchTab('alerts');
    }
}

/* SPARKLINE CHARTS */
let sparklineCharts = {};

function renderSparkline(canvasId, labels, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    if (sparklineCharts[canvasId]) {
        sparklineCharts[canvasId].destroy();
    }
    const ctx = canvas.getContext('2d');
    sparklineCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                borderColor: color,
                backgroundColor: color + '22',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

function loadSparklines() {
    const url = currentDevice ? `/api/history?device=${encodeURIComponent(currentDevice)}&points=20` : '/api/history?points=20';
    fetch(url)
        .then(r => r.json())
        .then(data => {
            renderSparkline('spark-temp', data.labels, data.temperature, '#f97316');
            renderSparkline('spark-vib', data.labels, data.vibration, '#a855f7');
            renderSparkline('spark-volt', data.labels, data.voltage, '#38bdf8');
            renderSparkline('spark-rpm', data.labels, data.rpm, '#22c55e');
        })
        .catch(error => console.error('Error loading sparklines:', error));
}

/* REPORT GENERATOR */
function openReportModal() {
    const list = document.getElementById('motor-picker-list');
    list.innerHTML = '';
    fetch('/api/devices')
        .then(r => r.json())
        .then(data => {
            data.devices.forEach(id => {
                const item = document.createElement('label');
                item.className = 'motor-picker-item';
                item.innerHTML = `
                    <input type="checkbox" class="round-check report-motor-check" value="${id}">
                    <span>${id}</span>
                `;
                list.appendChild(item);
            });
        });
    document.getElementById('report-modal').classList.add('open');
}

function closeReportModal() {
    document.getElementById('report-modal').classList.remove('open');
}

function toggleAllMotors(checked) {
    document.querySelectorAll('.report-motor-check').forEach(cb => cb.checked = checked);
}

function submitReport() {
    const motors = Array.from(document.querySelectorAll('.report-motor-check:checked')).map(cb => cb.value);
    const fields = Array.from(document.querySelectorAll('.report-field:checked')).map(cb => cb.value);
    const includePredictions = document.getElementById('report-predictions').checked;

    if (motors.length === 0) {
        alert("Please select at least one motor.");
        return;
    }
    if (fields.length === 0) {
        alert("Please select at least one field to include.");
        return;
    }

    fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ motors, fields, include_predictions: includePredictions })
    })
    .then(r => r.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'winteq_maintenance_report.pdf';
        a.click();
        closeReportModal();
    })
    .catch(error => console.error('Error generating report:', error));
}
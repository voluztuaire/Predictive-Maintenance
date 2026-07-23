let sensorChartInstance = null;
let sensorChartFullInstance = null;
let motorsCache = [];
let logsCache = [];
let currentDevice = null;
let expandedParam = null;
let fullSparkCharts = {};

const PARAM_DETAIL_CONFIG = {
    temp: { title: 'Temperature', fields: [
        { key: 'temperature', label: 'Temperature', color: '#f97316' }
    ], thresholdParams: ['Temperature'] },
    vib: { title: 'Vibration', fields: [
        { key: 'vibration_x', label: 'X', color: '#a855f7' },
        { key: 'vibration_y', label: 'Y', color: '#9333ea' },
        { key: 'vibration_z', label: 'Z', color: '#7e22ce' }
    ], thresholdParams: ['Vibration_X', 'Vibration_Y', 'Vibration_Z'] },
    volt: { title: 'Voltage', fields: [
        { key: 'voltage_l1', label: 'L1', color: '#38bdf8' },
        { key: 'voltage_l2', label: 'L2', color: '#0ea5e9' },
        { key: 'voltage_l3', label: 'L3', color: '#0369a1' }
    ], thresholdParams: ['Voltage_L1', 'Voltage_L2', 'Voltage_L3'] },
    current: { title: 'Current', fields: [
        { key: 'current_l1', label: 'L1', color: '#eab308' },
        { key: 'current_l2', label: 'L2', color: '#ca8a04' },
        { key: 'current_l3', label: 'L3', color: '#a16207' }
    ], thresholdParams: ['Current_L1', 'Current_L2', 'Current_L3'] },
    rpm: { title: 'Rotational Speed', fields: [
        { key: 'rpm', label: 'RPM', color: '#22c55e' }
    ], thresholdParams: ['Rotational_Speed'] },
    freq: { title: 'Frequency', fields: [
        { key: 'frequency', label: 'Frequency (Hz)', color: '#8b5cf6' }
    ], thresholdParams: ['Frequency'] },
    pf: { title: 'Power Factor', fields: [
        { key: 'power_factor', label: 'Power Factor', color: '#64748b' }
    ], thresholdParams: ['Power_Factor'] }
};

let alarmRulesCache = null;
async function getAlarmRulesCached() {
    if (alarmRulesCache) return alarmRulesCache;
    const res = await fetch('/api/alarm-rules');
    alarmRulesCache = await res.json();
    return alarmRulesCache;
}

function buildRangeText(rules, thresholdParams, deviceId) {
    const matched = rules.filter(r =>
        thresholdParams.includes(r.parameter) &&
        r.enabled &&
        (r.device === 'All' || r.device === deviceId)
    );
    if (matched.length === 0) return 'Range: --';

    const byTier = {};
    matched.forEach(r => {
        if (!byTier[r.tier]) byTier[r.tier] = r;
    });

    const parts = [];
    ['warning', 'critical', 'failure'].forEach(tier => {
        if (byTier[tier]) {
            const r = byTier[tier];
            const symbol = r.condition === 'more_than' ? '>' : '<';
            parts.push(`${tier[0].toUpperCase() + tier.slice(1)} ${symbol}${r.value}`);
        }
    });
    return parts.length ? parts.join(' &nbsp;&middot;&nbsp; ') : 'Range: --';
}

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    loadDevices();
    loadMotors();
    loadSensors();
    loadLogs();
});

/* THEME & MOBILE MENU */
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-mode');
        const icon = document.getElementById('theme-icon');
        if(icon) icon.classList.replace('fa-sun', 'fa-moon');
    }
}

function toggleTheme() {
    const body = document.body;
    const icon = document.getElementById('theme-icon');
    body.classList.toggle('light-mode');
    if (body.classList.contains('light-mode')) {
        localStorage.setItem('theme', 'light');
        if(icon) icon.classList.replace('fa-sun', 'fa-moon');
    } else {
        localStorage.setItem('theme', 'dark');
        if(icon) icon.classList.replace('fa-moon', 'fa-sun');
    }
    updateChartsTheme();
}

function toggleMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('mobile-overlay');
    if (sidebar && overlay) {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('open');
    }
}

function updateChartsTheme() {
    const isLight = document.body.classList.contains('light-mode');
    const gridColor = getChartColors().gridColor;
    const textColor = getChartColors().textColor;

    if (sensorChartInstance) {
        if(sensorChartInstance.options.plugins && sensorChartInstance.options.plugins.legend) {
            sensorChartInstance.options.plugins.legend.labels.color = textColor;
        }
        if(sensorChartInstance.options.scales.yTemp) {
            sensorChartInstance.options.scales.yTemp.grid.color = gridColor;
        }
        if(sensorChartInstance.options.scales.x) {
            sensorChartInstance.options.scales.x.grid.color = gridColor;
            sensorChartInstance.options.scales.x.ticks.color = textColor;
        }
        sensorChartInstance.update();
    }
    if (sensorChartFullInstance) {
        if(sensorChartFullInstance.options.plugins && sensorChartFullInstance.options.plugins.legend) {
            sensorChartFullInstance.options.plugins.legend.labels.color = textColor;
        }
        if(sensorChartFullInstance.options.scales.y) {
            sensorChartFullInstance.options.scales.y.grid.color = gridColor;
            sensorChartFullInstance.options.scales.y.ticks.color = textColor;
        }
        if(sensorChartFullInstance.options.scales.x) {
            sensorChartFullInstance.options.scales.x.grid.color = gridColor;
            sensorChartFullInstance.options.scales.x.ticks.color = textColor;
        }
        sensorChartFullInstance.update();
    }

    if (typeof forecastCompareCharts !== 'undefined') {
        Object.values(forecastCompareCharts).forEach(chart => {
            if (chart.options.scales.x) {
                chart.options.scales.x.grid.color = gridColor;
                chart.options.scales.x.ticks.color = textColor;
            }
            if (chart.options.scales.y) {
                chart.options.scales.y.grid.color = gridColor;
                chart.options.scales.y.ticks.color = textColor;
            }
            chart.update();
        });
    }
}


function getChartColors() {
    const isLight = document.body.classList.contains('light-mode');
    return {
        gridColor: isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)',
        textColor: isLight ? '#64748b' : '#9ca3af'
    };
}

/* CUSTOM MODAL */
function showModal(options) {
    const modal = document.getElementById('custom-modal');
    const icon = document.getElementById('modal-icon');
    const title = document.getElementById('modal-title');
    const message = document.getElementById('modal-message');
    const confirmBtn = document.getElementById('modal-confirm-btn');
    const cancelBtn = document.getElementById('modal-cancel-btn');

    const type = options.type || 'info';
    const iconMap = {
        success: 'fa-circle-check',
        error: 'fa-circle-xmark',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info',
        confirm: 'fa-question-circle'
    };
    const titleMap = {
        success: 'Success',
        error: 'Error',
        warning: 'Warning',
        info: 'Information',
        confirm: 'Confirm'
    };

    icon.className = `fa-solid ${iconMap[type] || iconMap.info} ${type}`;
    title.textContent = options.title || titleMap[type] || 'Notification';
    message.textContent = options.message || 'Operation completed.';

    if (type === 'confirm' || options.confirm) {
        confirmBtn.style.display = 'flex';
        cancelBtn.style.display = 'flex';
        confirmBtn.textContent = options.confirmText || 'Confirm';
        cancelBtn.textContent = options.cancelText || 'Cancel';
        
        const newConfirm = confirmBtn.cloneNode(true);
        const newCancel = cancelBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirm, confirmBtn);
        cancelBtn.parentNode.replaceChild(newCancel, cancelBtn);
        
        newConfirm.addEventListener('click', () => {
            closeModal();
            if (options.onConfirm) options.onConfirm();
        });
        newCancel.addEventListener('click', () => {
            closeModal();
            if (options.onCancel) options.onCancel();
        });
    } else {
        confirmBtn.style.display = 'flex';
        cancelBtn.style.display = 'none';
        confirmBtn.textContent = options.buttonText || 'OK';
        
        const newConfirm = confirmBtn.cloneNode(true);
        confirmBtn.parentNode.replaceChild(newConfirm, confirmBtn);
        
        newConfirm.addEventListener('click', () => {
            closeModal();
            if (options.onClose) options.onClose();
        });
    }

    modal.classList.add('open');
}

function closeModal() {
    document.getElementById('custom-modal').classList.remove('open');
}

document.addEventListener('DOMContentLoaded', function() {
    const modal = document.getElementById('custom-modal');
    if (modal) {
        modal.addEventListener('click', function(e) {
            if (e.target === this) {
                closeModal();
            }
        });
    }
});

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeModal();
    }
});

function chartOptions() {
    const colors = getChartColors();
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: colors.textColor, boxWidth: 12, padding: 16 } } },
        scales: {
            y: { grid: { color: colors.gridColor }, ticks: { color: colors.textColor }, beginAtZero: false },
            x: { grid: { color: colors.gridColor }, ticks: { color: colors.textColor } }
        }
    };
}

function initChart() {
    const url = currentDevice ? `/api/history?device=${encodeURIComponent(currentDevice)}&points=20` : '/api/history?points=20';
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const ctx = document.getElementById('sensorChart').getContext('2d');
            if (sensorChartInstance) sensorChartInstance.destroy();
            sensorChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [
                        { label: 'Temperature (C)', data: data.temperature, borderColor: '#f97316', backgroundColor: 'rgba(249, 115, 22, 0.1)', borderWidth: 2, tension: 0.4, fill: true, yAxisID: 'yTemp' },
                        { label: 'Voltage (V)', data: data.voltage, borderColor: '#38bdf8', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'yVolt' },
                        { label: 'Current (A)', data: data.current, borderColor: '#eab308', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'yCurr' },
                        { label: 'Vibration (mm/s)', data: data.vibration, borderColor: '#ec4899', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'yVib' },
                        { label: 'RPM', data: data.rpm, borderColor: '#22c55e', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'yRPM' },
                        { label: 'Frequency (Hz)', data: data.frequency, borderColor: '#8b5cf6', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'yFreq' },
                        { label: 'Power Factor', data: data.power_factor, borderColor: '#64748b', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'yPF' }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: getChartColors().textColor, boxWidth: 12, padding: 16 } } },
                    scales: {
                        yTemp: { position: 'left', grid: { color: getChartColors().gridColor }, ticks: { color: '#f97316' }, title: { display: true, text: 'Temp (C)', color: '#f97316' } },
                        yVolt: { position: 'right', grid: { display: false }, ticks: { color: '#38bdf8' }, title: { display: true, text: 'Voltage (V)', color: '#38bdf8' }, min: 380, max: 410 },
                        yCurr: { position: 'right', display: false, min: 0, max: 12 },
                        yVib: { position: 'right', display: false, min: 0, max: 10 },
                        yRPM: { position: 'right', display: false, min: 1400, max: 1550 },
                        yFreq: { position: 'right', display: false, min: 45, max: 55 },
                        yPF: { position: 'right', display: false, min: 0.7, max: 1.0 },
                        x: { grid: { color: getChartColors().gridColor }, ticks: { color: getChartColors().textColor } }
                    }
                }
            });
        })
        .catch(error => console.error('Error loading main chart:', error));
}

function initFullChart() {
    const url = currentDevice ? `/api/history?device=${encodeURIComponent(currentDevice)}&points=30` : '/api/history?points=30';
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const canvas = document.getElementById('sensorChartFull');
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (sensorChartFullInstance) sensorChartFullInstance.destroy();
            sensorChartFullInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [
                        { label: 'Temperature (C)', data: data.temperature, borderColor: '#f97316', yAxisID: 'y', borderWidth: 2, tension: 0.4, fill: true },
                        { label: 'Vibration (mm/s)', data: data.vibration, borderColor: '#a855f7', yAxisID: 'y', borderWidth: 2, tension: 0.4, fill: false },
                        { label: 'Voltage (V)', data: data.voltage, borderColor: '#38bdf8', yAxisID: 'y1', borderWidth: 2, tension: 0.4, fill: false },
                        { label: 'Frequency (Hz)', data: data.frequency, borderColor: '#8b5cf6', yAxisID: 'y1', borderWidth: 2, tension: 0.4, fill: false },
                        { label: 'Power Factor', data: data.power_factor, borderColor: '#64748b', yAxisID: 'y1', borderWidth: 2, tension: 0.4, fill: false }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: getChartColors().textColor } } },
                    scales: {
                        y: { position: 'left', ticks: { color: getChartColors().textColor }, grid: { color: getChartColors().gridColor } },
                        y1: { position: 'right', ticks: { color: '#38bdf8' }, grid: { display: false } },
                        x: { ticks: { color: getChartColors().textColor }, grid: { color: getChartColors().gridColor } }
                    }
                }
            });
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
        updateDetailHeader(paramKey);
        renderFullSpark(paramKey);
    } else {
        expandedParam = null;
    }
}

async function updateDetailHeader(paramKey) {
    const config = PARAM_DETAIL_CONFIG[paramKey];
    if (!config) return;

    const titleEl = document.getElementById('detail-title-' + paramKey);
    const rangeEl = document.getElementById('detail-range-' + paramKey);
    if (titleEl) titleEl.textContent = 'Detail ' + config.title;

    if (rangeEl) {
        rangeEl.innerHTML = 'Loading range...';
        try {
            const rules = await getAlarmRulesCached();
            rangeEl.innerHTML = buildRangeText(rules, config.thresholdParams, currentDevice);
        } catch (e) {
            rangeEl.innerHTML = 'Range: --';
        }
    }
}

function renderFullSpark(paramKey) {
    const url = currentDevice ? `/api/history?device=${encodeURIComponent(currentDevice)}&points=30` : '/api/history?points=30';
    fetch(url)
        .then(r => r.json())
        .then(data => {
            const config = PARAM_DETAIL_CONFIG[paramKey];
            if (!config) return;
            const canvasId = 'spark-' + paramKey + '-full';

            if (fullSparkCharts[canvasId]) fullSparkCharts[canvasId].destroy();

            const datasets = config.fields.map(f => ({
                label: f.label,
                data: data[f.key],
                borderColor: f.color,
                backgroundColor: f.color + '22',
                borderWidth: 2,
                tension: 0.4,
                fill: config.fields.length === 1,
                pointRadius: 3
            }));

            const ctx = document.getElementById(canvasId).getContext('2d');
            fullSparkCharts[canvasId] = new Chart(ctx, {
                type: 'line',
                data: { labels: data.labels, datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: config.fields.length > 1, labels: { color: getChartColors().textColor, boxWidth: 12 } } },
                    scales: {
                        x: { grid: { color: getChartColors().gridColor }, ticks: { color: getChartColors().textColor } },
                        y: { grid: { color: getChartColors().gridColor }, ticks: { color: getChartColors().textColor }, beginAtZero: false }
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
            fetchAndRenderAlerts();
        })
        .catch(error => console.error('Error loading devices:', error));
}

function fetchAndUpdateMetrics() {
    const url = currentDevice ? `/api/status?device=${encodeURIComponent(currentDevice)}` : '/api/status';

    fetch(url)
        .then(response => response.json())
        .then(data => {
            document.getElementById('val-health').innerText = data.health_score + "%";
            document.getElementById('val-rul').innerHTML = Math.round(data.rul_hours).toLocaleString() + ' <span class="unit">Hrs</span>';
            document.getElementById('val-falsealarm').innerText = data.false_alarm_rate + "%";
            document.getElementById('val-failprob').innerText = data.failure_probability + "%";
            document.getElementById('val-failprob-window').innerText = data.risk_window_label || 'Based on current sensor pattern';
            document.getElementById('val-risklevel').innerText = data.failure_probability + "%";
            // Update risk gauge rotation
            const riskDeg = Math.round((data.failure_probability / 100) * 360);
            document.getElementById('risk-gauge').style.setProperty('--risk-deg', `${riskDeg}deg`);
            
            // Degradation list
            if (data.degradation_list) {
                const degList = document.getElementById('val-degradation-list');
                if (degList) {
                    degList.innerHTML = data.degradation_list.map((d, i) => 
                        `<li><span class="dot ${i === 0 ? 'primary' : 'secondary'}"></span> ${d.label} <span class="dot-value">${d.value}</span></li>`
                    ).join('');
                }
            }
            // AI Confidence
            if (data.confidence && document.getElementById('val-ai-confidence')) {
                document.getElementById('val-ai-confidence').innerText = data.confidence;
            }

            document.getElementById('val-temperature').innerHTML = data.temperature + ' <span class="unit">C</span>';
            document.getElementById('val-pressure').innerHTML = Math.round(data.pressure) + ' <span class="unit">RPM</span>';
            document.getElementById('val-recommendation').innerText = data.recommendation || 'System operating within normal parameters.';
            document.getElementById('val-frequency').innerHTML = data.frequency + ' <span class="unit">Hz</span>';
            document.getElementById('val-powerfactor').innerText = data.power_factor;

            // BARU: per-fase
            document.getElementById('val-voltage-l1').innerText = data.voltage_l1 + ' V';
            document.getElementById('val-voltage-l2').innerText = data.voltage_l2 + ' V';
            document.getElementById('val-voltage-l3').innerText = data.voltage_l3 + ' V';

            document.getElementById('val-current-l1').innerText = data.current_l1 + ' A';
            document.getElementById('val-current-l2').innerText = data.current_l2 + ' A';
            document.getElementById('val-current-l3').innerText = data.current_l3 + ' A';

            document.getElementById('val-vib-x').innerText = data.vibration_x + ' mm/s';
            document.getElementById('val-vib-y').innerText = data.vibration_y + ' mm/s';
            document.getElementById('val-vib-z').innerText = data.vibration_z + ' mm/s';

            const healthNote = document.getElementById('val-health-note');
            const healthIcon = document.querySelector('.kpi-icon.health');
            
            // Clear any inline styles from previous states
            healthNote.style.color = '';
            if (healthIcon) {
                healthIcon.style.color = '';
                healthIcon.style.background = '';
            }

            if (data.health_score >= 70) {
                healthNote.innerText = 'Normal operating range';
                healthNote.className = 'card-delta positive';
            } else if (data.health_score >= 40) {
                healthNote.innerText = 'Reduced performance detected';
                healthNote.className = 'card-delta text-warning';
                if (healthIcon) {
                    healthIcon.style.color = 'var(--warning)';
                    healthIcon.style.background = 'rgba(245, 158, 11, 0.12)';
                }
            } else {
                healthNote.innerText = 'Immediate attention required';
                healthNote.className = 'card-delta text-danger';
                if (healthIcon) {
                    healthIcon.style.color = 'var(--danger)';
                    healthIcon.style.background = 'rgba(239, 68, 68, 0.12)';
                }
            }

            currentDevice = data.device;
            loadSparklines();
            initChart();
            
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
                    <td>${s.vibration_x}</td>
                    <td>${s.vibration_y}</td>
                    <td>${s.vibration_z}</td>
                    <td>${s.voltage_l1}</td>
                    <td>${s.voltage_l2}</td>
                    <td>${s.voltage_l3}</td>
                    <td>${s.current_l1}</td>
                    <td>${s.current_l2}</td>
                    <td>${s.current_l3}</td>
                    <td>${Math.round(s.pressure)}</td>
                    <td>${s.frequency}</td>
                    <td>${s.power_factor}</td>
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

function saveConfiguration() {
    showModal({
        type: 'success',
        title: 'Configuration Saved',
        message: 'AI model configuration has been updated successfully.',
        buttonText: 'OK'
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

    const pages = ['dashboard', 'forecast', 'assets', 'sensors', 'alerts', 'condition', 'training', 'settings'];
    pages.forEach(page => {
        const el = document.getElementById('page-' + page);
        if (el) el.style.display = (page === tabName) ? 'flex' : 'none';
    });

    if (tabName === 'forecast') { loadForecastComparePage(); }
    if (tabName === 'condition') { loadConditionAlerts(); }
    if (tabName === 'training') { loadReviewQueue(); loadModelHistory(); loadPendingTrainingData(); }
    if (tabName === 'settings') { loadAlarmRules(); }
}

function selectDevice(deviceId) {
    currentDevice = deviceId;
    fetchAndUpdateMetrics();
    fetchAndRenderAlerts();
    loadSensors();
    loadLogs();
    const forecastPage = document.getElementById('page-forecast');
    if (forecastPage && forecastPage.style.display !== 'none') {
        loadForecastComparePage();
    }
}

function refreshData() {
    fetchAndUpdateMetrics();
    fetchAndRenderAlerts();
}

function triggerBell() {
    fetch('/api/notifications')
        .then(r => r.json())
        .then(data => {
            if (data.notifications.length === 0) {
                showModal({
                    type: 'info',
                    title: 'Notifications',
                    message: 'You have no new notifications at this time.',
                    buttonText: 'Got it'
                });
            } else {
                showNotificationModal(data.notifications.slice(0, 10));
            }
            fetch('/api/notifications/read', { method: 'POST' });
        })
        .catch(err => {
            showModal({
                type: 'error',
                title: 'Notifications',
                message: 'Could not load notifications.',
                buttonText: 'OK'
            });
            console.error('Error loading notifications:', err);
        });
}

function severityIconBox(severity) {
    const map = {
        Warning: { icon: 'fa-triangle-exclamation', cls: 'warning' },
        Critical: { icon: 'fa-circle-exclamation', cls: 'warning' },
        Failure: { icon: 'fa-circle-exclamation', cls: 'critical' }
    };
    return map[severity] || { icon: 'fa-circle-info', cls: 'info' };
}

function showNotificationModal(notifications) {
    const modal = document.getElementById('custom-modal');
    const icon = document.getElementById('modal-icon');
    const title = document.getElementById('modal-title');
    const message = document.getElementById('modal-message');
    const confirmBtn = document.getElementById('modal-confirm-btn');
    const cancelBtn = document.getElementById('modal-cancel-btn');

    icon.className = 'fa-solid fa-triangle-exclamation warning';
    title.textContent = 'Notifications';
    cancelBtn.style.display = 'none';
    confirmBtn.style.display = 'flex';
    confirmBtn.textContent = 'Close';

    const newConfirm = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirm, confirmBtn);
    newConfirm.addEventListener('click', closeModal);

    message.innerHTML = notifications.map(n => {
        const meta = severityIconBox(n.severity);
        return `
            <div class="notif-item">
                <div class="icon-box ${meta.cls}"><i class="fa-solid ${meta.icon}"></i></div>
                <div class="notif-body">
                    <div class="notif-title">${n.title}</div>
                    <div class="notif-desc">${n.description}</div>
                    <div class="notif-time">${n.time}</div>
                </div>
            </div>
        `;
    }).join('');

    modal.classList.add('open');
}

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

let sparklineCharts = {};

function renderSparkline(canvasId, labels, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    if (sparklineCharts[canvasId]) {
        const chart = sparklineCharts[canvasId];
        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        chart.update('none'); 
        return;
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
            scales: { x: { display: false }, y: { display: false, beginAtZero: false } }
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
            renderSparkline('spark-current', data.labels, data.current, '#eab308');
            renderSparkline('spark-rpm', data.labels, data.rpm, '#22c55e');
            renderSparkline('spark-freq', data.labels, data.frequency, '#8b5cf6');
            renderSparkline('spark-pf', data.labels, data.power_factor, '#64748b');
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
            updateMotorCount();
        });
    document.getElementById('report-modal').classList.add('open');
}

function closeReportModal() {
    document.getElementById('report-modal').classList.remove('open');
}

function toggleAllMotors(checked) {
    document.querySelectorAll('.report-motor-check').forEach(cb => {
        cb.checked = checked;
        updateMotorItemStyle(cb);
    });
    updateMotorCount();
}

function updateMotorItemStyle(checkbox) {
    const item = checkbox.closest('.motor-picker-item');
    if (checkbox.checked) {
        item.classList.add('selected');
    } else {
        item.classList.remove('selected');
    }
}

function updateMotorCount() {
    const checked = document.querySelectorAll('.report-motor-check:checked').length;
    const total = document.querySelectorAll('.report-motor-check').length;
    const countEl = document.getElementById('selected-count');
    const totalEl = document.getElementById('total-count');
    if (countEl) countEl.textContent = checked;
    if (totalEl) totalEl.textContent = `Total: ${total} motors`;
}

document.addEventListener('change', function(e) {
    if (e.target.classList.contains('round-check') && e.target.closest('.motor-picker-item')) {
        updateMotorItemStyle(e.target);
        updateMotorCount();
    }
});

function submitReport() {
    const btn = document.getElementById('report-generate-btn');
    const btnText = document.getElementById('report-btn-text');
    const spinner = document.getElementById('report-btn-spinner');
    
    const motors = Array.from(document.querySelectorAll('.report-motor-check:checked')).map(cb => cb.value);
    const fields = Array.from(document.querySelectorAll('.report-field:checked')).map(cb => cb.value);
    const includePredictions = document.getElementById('report-predictions').checked;

    if (motors.length === 0) {
        showModal({
            type: 'warning',
            title: 'No Motors Selected',
            message: 'Please select at least one motor to include in the report.',
            buttonText: 'OK'
        });
        return;
    }
    if (fields.length === 0) {
        showModal({
            type: 'warning',
            title: 'No Fields Selected',
            message: 'Please select at least one field to include in the report.',
            buttonText: 'OK'
        });
        return;
    }

    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'inline';

    fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ motors, fields, include_predictions: includePredictions })
    })
    .then(r => {
        if (!r.ok) {
            throw new Error('Failed to generate report');
        }
        return r.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'winteq_maintenance_report.pdf';
        a.click();
        closeReportModal();
        showModal({
            type: 'success',
            title: 'Report Generated',
            message: 'PDF report has been downloaded successfully.',
            buttonText: 'OK'
        });
    })
    .catch(error => {
        showModal({
            type: 'error',
            title: 'Error',
            message: 'Failed to generate report. Please try again.',
            buttonText: 'OK'
        });
        console.error('Error generating report:', error);
    })
    .finally(() => {
        btn.disabled = false;
        btnText.style.display = 'inline';
        spinner.style.display = 'none';
    });
}

let tickCounter = 0;

setInterval(() => {
    tickCounter++;
    fetch('/api/tick', { method: 'POST' }).then(() => {
        fetchAndUpdateMetrics(); 
        loadSensors();          
        
        if (tickCounter % 3 === 0) {
            loadMotors();
            loadLogs();
            fetchAndRenderAlerts();
            updateNotifBadge();
            const trainingPage = document.getElementById('page-training');
            if (trainingPage && trainingPage.style.display !== 'none') {
                loadReviewQueue();
                loadPendingTrainingData();
            }
            const conditionAlertsPage = document.getElementById('page-condition-alerts');
            if (conditionAlertsPage && conditionAlertsPage.style.display !== 'none') {
                loadConditionAlerts();
            }
            tickCounter = 0;
        }
    });
}, 30000);

function updateNotifBadge() {
    fetch('/api/notifications')
        .then(r => r.json())
        .then(data => {
            const bell = document.querySelector('.btn-icon[onclick="triggerBell()"]');
            if (!bell) return;

            let badge = bell.querySelector('.notif-badge');
            if (data.unread_count > 0) {
                if (!badge) {
                    badge = document.createElement('span');
                    badge.className = 'notif-badge';
                    bell.appendChild(badge);
                }
                badge.textContent = data.unread_count > 9 ? '9+' : data.unread_count;
            } else if (badge) {
                badge.remove();
            }
        })
        .catch(err => console.error('Error checking notif badge:', err));
}

let forecastChartInstance = null;


// ============================================================
// CHATBOT WIDGET
// ============================================================
(function () {
    const toggleBtn = document.getElementById("chatbot-toggle-btn");
    const closeBtn = document.getElementById("chatbot-close-btn");
    const widget = document.getElementById("chatbot-widget");
    const messagesEl = document.getElementById("chatbot-messages");
    const inputEl = document.getElementById("chatbot-input");
    const sendBtn = document.getElementById("chatbot-send-btn");

    function openChat() {
        widget.classList.add("open");
        toggleBtn.classList.add("hidden");
        if (messagesEl.children.length === 0) {
            addBubble("bot", "Hi, I'm your maintenance assistant. Ask me about any motor's status, or which motors need attention today.");
        }
        inputEl.focus();
    }

    function closeChat() {
        widget.classList.remove("open");
        toggleBtn.classList.remove("hidden");
    }

    function addBubble(role, text) {
        const div = document.createElement("div");
        div.className = "chat-bubble " + role;
        div.innerHTML = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
            
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
    }

    async function sendMessage() {
        const text = inputEl.value.trim();
        if (!text) return;

        addBubble("user", text);
        inputEl.value = "";
        sendBtn.disabled = true;

        const typingBubble = addBubble("bot typing", "Thinking...");

        try {
            const res = await fetch("/api/chat/llm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            typingBubble.remove();
            addBubble("bot", data.response || "No response received.");
        } catch (err) {
            typingBubble.remove();
            addBubble("bot", "Error: could not reach the assistant. Make sure Ollama is running.");
        } finally {
            sendBtn.disabled = false;
        }
    }

    toggleBtn.addEventListener("click", openChat);
    closeBtn.addEventListener("click", closeChat);
    sendBtn.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendMessage();
    });
})();

let forecastCompareCharts = {};

function combineVibrationRMS(sensors) {
    return sensors.Vibration_X.map((vx, i) => {
        const vy = sensors.Vibration_Y[i];
        const vz = sensors.Vibration_Z[i];
        return Math.sqrt(vx*vx + vy*vy + vz*vz);
    });
}

function combineVoltageAvg(sensors) {
    return sensors.Voltage_L1.map((v1, i) => {
        const v2 = sensors.Voltage_L2[i];
        const v3 = sensors.Voltage_L3[i];
        return (v1 + v2 + v3) / 3;
    });
}

function combineCurrentAvg(sensors) {
    return sensors.Current_L1.map((c1, i) => {
        const c2 = sensors.Current_L2[i];
        const c3 = sensors.Current_L3[i];
        return (c1 + c2 + c3) / 3;
    });
}

function loadForecastComparePage() {
    const grid = document.getElementById('forecast-compare-grid');
    
    grid.innerHTML = `
        <div style="text-align: center; padding: 60px 20px; color: var(--primary-orange);">
            <i class="fa-solid fa-circle-notch fa-spin" style="font-size: 32px; margin-bottom: 16px;"></i>
            <p style="color: var(--text-muted); font-size: 14px;">Generating live AI forecast for the next 48 hours...</p>
        </div>
    `;

    const historyUrl = currentDevice ? `/api/history?device=${encodeURIComponent(currentDevice)}&points=30` : '/api/history?points=30';
    const forecastUrl = currentDevice ? `/api/forecast?device=${encodeURIComponent(currentDevice)}&horizon=48` : '/api/forecast?horizon=48';

    Promise.all([fetch(historyUrl).then(r => r.json()), fetch(forecastUrl).then(r => r.json())])
        .then(([hist, fcst]) => {
            grid.innerHTML = '';

            const params = [
                { key: 'temperature', label: 'Temperature (°C)', color: '#f97316', histData: hist.temperature, fcstData: fcst.sensors.Temperature },

                { key: 'vib_x', label: 'Vibration X (mm/s)', color: '#a855f7', histData: hist.vibration_x, fcstData: fcst.sensors.Vibration_X },
                { key: 'vib_y', label: 'Vibration Y (mm/s)', color: '#9333ea', histData: hist.vibration_y, fcstData: fcst.sensors.Vibration_Y },
                { key: 'vib_z', label: 'Vibration Z (mm/s)', color: '#7e22ce', histData: hist.vibration_z, fcstData: fcst.sensors.Vibration_Z },

                { key: 'volt_l1', label: 'Voltage L1 (V)', color: '#38bdf8', histData: hist.voltage_l1, fcstData: fcst.sensors.Voltage_L1 },
                { key: 'volt_l2', label: 'Voltage L2 (V)', color: '#0ea5e9', histData: hist.voltage_l2, fcstData: fcst.sensors.Voltage_L2 },
                { key: 'volt_l3', label: 'Voltage L3 (V)', color: '#0369a1', histData: hist.voltage_l3, fcstData: fcst.sensors.Voltage_L3 },

                { key: 'curr_l1', label: 'Current L1 (A)', color: '#eab308', histData: hist.current_l1, fcstData: fcst.sensors.Current_L1 },
                { key: 'curr_l2', label: 'Current L2 (A)', color: '#ca8a04', histData: hist.current_l2, fcstData: fcst.sensors.Current_L2 },
                { key: 'curr_l3', label: 'Current L3 (A)', color: '#a16207', histData: hist.current_l3, fcstData: fcst.sensors.Current_L3 },

                { key: 'rpm', label: 'Rotational Speed (RPM)', color: '#22c55e', histData: hist.rpm, fcstData: fcst.sensors.Rotational_Speed },
                { key: 'freq', label: 'Frequency (Hz)', color: '#8b5cf6', histData: hist.frequency, fcstData: fcst.sensors.Frequency },
                { key: 'pf', label: 'Power Factor', color: '#64748b', histData: hist.power_factor, fcstData: fcst.sensors.Power_Factor },
            ];

            params.forEach(p => {
                const block = document.createElement('div');
                block.className = 'forecast-param-block';
                block.innerHTML = `
                    <h4>${p.label}</h4>
                    <div class="forecast-param-chart"><canvas id="fcst-combo-${p.key}"></canvas></div>
                `;
                grid.appendChild(block);

                setTimeout(() => renderComboChart(`fcst-combo-${p.key}`, hist.labels, p.histData, fcst.labels, p.fcstData, p.color), 0);
            });
        })
        .catch(err => {
            console.error('Error loading forecast compare page:', err);
            grid.innerHTML = '<div style="text-align:center; padding: 20px; color: var(--danger);">Failed to load forecast data.</div>';
        });
}

function renderComboChart(canvasId, histLabels, histData, fcstLabels, fcstData, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    if (forecastCompareCharts[canvasId]) forecastCompareCharts[canvasId].destroy();

    // Gabung label jadi 1 sumbu X kontinu
    const labels = [...histLabels, ...fcstLabels];

    // Dataset historical dan forecast
    const historicalSeries = [...histData, ...new Array(fcstData.length).fill(null)];
    const forecastSeries = [
        ...new Array(histData.length - 1).fill(null),
        histData[histData.length - 1],
        ...fcstData
    ];

    // --- FITUR BARU: Plugin Custom untuk Garis "Sekarang" ---
    // Titik "sekarang" ada di index terakhir data historis
    const currentIndex = histData.length - 1; 

    const currentLinePlugin = {
        id: 'currentLine',
        afterDraw(chart) {
            const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
            const xPos = x.getPixelForValue(currentIndex);
            const isLight = document.body.classList.contains('light-mode');
            const lineColor = isLight ? 'rgba(0, 0, 0, 0.35)' : 'rgba(255, 255, 255, 0.4)';
            const textColor = isLight ? 'rgba(0, 0, 0, 0.55)' : 'rgba(255, 255, 255, 0.6)';

            ctx.save();

            ctx.beginPath();
            ctx.strokeStyle = lineColor;
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.moveTo(xPos, top);
            ctx.lineTo(xPos, bottom);
            ctx.stroke();

            ctx.fillStyle = '#f97316';
            ctx.beginPath();
            ctx.arc(xPos, top, 4, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = textColor;
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText('CURRENT', xPos, top - 6);

            ctx.restore();
        }
    };
    // --------------------------------------------------------

    const ctx = canvas.getContext('2d');
    forecastCompareCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Historical',
                    data: historicalSeries,
                    borderColor: color,
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Forecast',
                    data: forecastSeries,
                    borderColor: color,
                    borderDash: [6, 4],
                    borderWidth: 2,
                    tension: 0.35,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            layout: {
                padding: { top: 20 } // Kasih ruang ekstra di atas biar teks "CURRENT" gak kepotong
            },
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: { grid: { color: getChartColors().gridColor }, ticks: { color: getChartColors().textColor, maxTicksLimit: 8 } },
                y: { grid: { color: getChartColors().gridColor }, ticks: { color: getChartColors().textColor } }
            }
        },
        // Masukkan plugin custom-nya ke sini
        plugins: [currentLinePlugin] 
    });
}

function loadConditionAlerts() {
    fetch('/api/threshold-alerts')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('condition-alerts-list');
            container.innerHTML = '';

            if (data.alerts.length === 0) {
                container.innerHTML = '<p style="color: var(--text-muted); font-size: 12px; padding: 12px 0;">No threshold violations detected across monitored motors.</p>';
                return;
            }

            data.alerts.forEach(a => {
                const violList = a.violations.map(v =>
                    `<span class="severity-tag ${v.tier}">${v.parameter}: ${v.actual_value} (thr ${v.threshold})</span>`
                ).join(' ');

                const row = document.createElement('div');
                row.className = 'log-row';
                row.innerHTML = `
                    <div class="log-info">
                        <div class="icon-box ${a.status_color === 'red' ? 'critical' : a.status_color === 'orange' ? 'warning' : 'info'}">
                            <i class="fa-solid fa-gauge-high"></i>
                        </div>
                        <div>
                            <h4>${a.condition_label} — ${a.motor_id}</h4>
                            <div class="viol-list-wrapper">${violList}</div>
                            <div class="log-meta">${a.timestamp}</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 10px; flex-shrink: 0;">
                    <span class="severity-tag ${a.condition_label.toLowerCase()}">${a.total_violations} violation(s)</span>
                            <!-- Tombol baru di bawah ini -->
                            <button class="btn-action" onclick="submitForReview('${a.motor_id}', '${a.timestamp}')">Submit for Review</button>
                    </div>                
                `;
                container.appendChild(row);
            });
        })
        .catch(err => console.error('Error loading condition alerts:', err));
}

function submitForReview(deviceId, timestamp) {
    fetch(`/api/expert-review/submit`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device: deviceId, timestamp: timestamp })
    })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                showModal({ type: 'warning', title: 'Cannot Submit', message: data.error, buttonText: 'OK' });
            } else {
                showModal({ type: 'success', title: 'Submitted', message: `${deviceId} added to expert review queue.`, buttonText: 'OK' });
                loadConditionAlerts();
            }
        })
        .catch(err => console.error('Error submitting review:', err));
}

function loadReviewQueue() {
    fetch('/api/expert-review/list?status=pending')
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById('review-queue-list');
            container.innerHTML = '';

            if (data.length === 0) {
                container.innerHTML = '<p style="color: var(--text-muted); font-size: 12px; padding: 12px 0;">No pending reviews.</p>';
                return;
            }

            data.forEach(item => {
                const row = document.createElement('div');
                row.className = 'log-row';
                row.innerHTML = `
                    <div class="log-info">
                        <div class="icon-box warning"><i class="fa-solid fa-clipboard-question"></i></div>
                        <div>
                            <h4>${item.motor_id} — ${item.threshold_alert.condition_label}</h4>
                            <div class="viol-list-wrapper">
                                ${item.threshold_alert.violations.map(v =>
                                    `<span class="severity-tag ${v.tier}">${v.parameter}: ${v.actual_value} (thr ${v.threshold})</span>`
                                ).join('')}
                            </div>
                            <div class="log-meta">${item.created_at}</div>
                        </div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:8px; align-items:flex-end; flex-shrink:0; min-width:320px; max-width:450px;">
                        <div style="display:flex; gap:8px; width:100%; justify-content:flex-end;">
                            <select id="label-${item.review_id}" class="filter-select" style="flex:1;">
                                <option value="Normal">Normal</option>
                                <option value="Warning">Warning</option>
                                <option value="Critical" selected>Critical</option>
                                <option value="Failure">Failure</option>
                            </select>
                            <select id="fault-${item.review_id}" class="filter-select" style="flex:1;">
                                <option value="Normal">Normal</option>
                                <option value="Rotor Bar">Rotor Bar</option>
                                <option value="Bearing Wear">Bearing Wear</option>
                                <option value="Misalignment">Misalignment</option>
                                <option value="Stator Winding">Stator Winding</option>
                                <option value="Other">Other / Unrecognized</option>
                            </select>
                        </div>
                        <input type="text" id="notes-${item.review_id}" placeholder="Specify anomaly details (Optional)" class="setting-input" style="width: 100%; padding: 6px 12px; height: 32px; font-size: 13px; box-sizing: border-box;">
                        <div style="display:flex; gap:8px; justify-content:flex-end; width:100%;">
                            <button class="btn-action" onclick="approveReview('${item.review_id}')">Approve</button>
                            <button class="btn-outline" onclick="rejectReview('${item.review_id}')">Reject</button>
                        </div>
                    </div>
                `;
                container.appendChild(row);
            });
        })
        .catch(err => console.error('Error loading review queue:', err));
}

function approveReview(reviewId) {
    const label = document.getElementById(`label-${reviewId}`).value;
    const fault = document.getElementById(`fault-${reviewId}`).value;
    const notesInput = document.getElementById(`notes-${reviewId}`);
    const notes = notesInput ? notesInput.value : '';

    fetch('/api/expert-review/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_id: reviewId, expert_label: label, expert_fault_type: fault, notes: notes })
    })
    .then(r => r.json())
    .then(() => {
        showModal({ type: 'success', title: 'Approved', message: 'Label saved to training dataset.', buttonText: 'OK' });
        loadReviewQueue();
        loadPendingTrainingData();
    })
    .catch(err => console.error('Error approving review:', err));
}

function rejectReview(reviewId) {
    fetch('/api/expert-review/reject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review_id: reviewId })
    })
    .then(r => r.json())
    .then(() => {
        showModal({ type: 'info', title: 'Rejected', message: 'Review discarded, not added to training data.', buttonText: 'OK' });
        loadReviewQueue();
    })
    .catch(err => console.error('Error rejecting review:', err));
}

function loadPendingTrainingData() {
    fetch('/api/admin/training-data/pending')
        .then(r => r.json())
        .then(data => {
            document.getElementById('pending-count').textContent = data.count;
            const body = document.getElementById('pending-training-body');
            body.innerHTML = '';
            if (data.count === 0) {
                body.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:var(--text-muted);">No approved expert data yet.</td></tr>';
                return;
            }
            data.rows.forEach(r => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${r.Timestamp || '-'}</td>
                    <td>${r.Motor_ID || '-'}</td>
                    <td>${r.Temperature ?? '-'}</td>
                    <td>${r.Vibration_X ?? '-'}</td>
                    <td><span class="status-pill ${r.Motor_State}"><span class="dot"></span>${r.Motor_State}</span></td>
                    <td>${r.Fault_Type_True || '-'}</td>
                    <td>${r.Expert_ID || '-'}</td>
                    <td>${r.Reviewed_At ? new Date(r.Reviewed_At).toLocaleString() : '-'}</td>
                `;
                body.appendChild(tr);
            });
        })
        .catch(err => console.error('Error loading pending training data:', err));
}

function openSnapshotModal(version) {
    fetch(`/api/admin/models/${version}/training-data`)
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                showModal({ type: 'warning', title: 'No Snapshot', message: data.error, buttonText: 'OK' });
                return;
            }
            document.getElementById('snapshot-modal-title').textContent = `Training Data Used — v${version} (${data.count} rows)`;
            const head = document.getElementById('snapshot-table-head');
            const body = document.getElementById('snapshot-table-body');
            head.innerHTML = '';
            body.innerHTML = '';
            
            if (data.rows.length > 0) {
                const preferredOrder = [
                    'Timestamp', 'Motor_ID', 
                    'Voltage_L1', 'Voltage_L2', 'Voltage_L3', 
                    'Current_L1', 'Current_L2', 'Current_L3', 
                    'Frequency', 'Power_Factor', 
                    'Temperature', 'Vibration_X', 'Vibration_Y', 'Vibration_Z', 
                    'Rotational_Speed', 'Motor_State', 'Fault_Type_True'
                ];
                const actualKeys = Object.keys(data.rows[0]);
                const keys = preferredOrder.filter(k => actualKeys.includes(k));
                actualKeys.forEach(k => {
                    if (!keys.includes(k)) keys.push(k);
                });

                const headTr = document.createElement('tr');
                keys.forEach(k => {
                    const th = document.createElement('th');
                    th.textContent = k.replace(/_/g, ' ');
                    headTr.appendChild(th);
                });
                head.appendChild(headTr);

                // Show the last 100 rows, reversed (newest first)
                const displayRows = data.rows.slice(-100).reverse();
                displayRows.forEach(r => { 
                    const tr = document.createElement('tr');
                    keys.forEach(k => {
                        const td = document.createElement('td');
                        td.textContent = r[k] !== null && r[k] !== undefined ? r[k] : '-';
                        tr.appendChild(td);
                    });
                    body.appendChild(tr);
                });
            }
            document.getElementById('snapshot-modal').classList.add('open');
        })
        .catch(err => console.error('Error loading snapshot:', err));
}

function closeSnapshotModal() {
    document.getElementById('snapshot-modal').classList.remove('open');
}

function triggerRetrain() {
    const resultBox = document.getElementById('retrain-result');
    resultBox.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Retraining in progress, this may take a minute...';

    fetch('/api/admin/retrain', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'started' || data.status === 'already_running') {
                // Start polling
                const pollInterval = setInterval(() => {
                    fetch('/api/admin/retrain/status')
                        .then(r => r.json())
                        .then(statusData => {
                            if (statusData.status === 'completed') {
                                clearInterval(pollInterval);
                                renderRetrainResult(statusData.result, resultBox);
                            } else if (statusData.status === 'error') {
                                clearInterval(pollInterval);
                                resultBox.innerHTML = '<span style="color:var(--danger);">Retrain failed: ' + statusData.error + '</span>';
                            }
                        })
                        .catch(err => {
                            clearInterval(pollInterval);
                            resultBox.innerHTML = '<span style="color:var(--danger);">Polling failed: ' + err + '</span>';
                        });
                }, 3000);
            } else {
                resultBox.innerHTML = '<span style="color:var(--danger);">Unexpected response: ' + JSON.stringify(data) + '</span>';
            }
        })
        .catch(err => {
            resultBox.innerHTML = '<span style="color:var(--danger);">Failed to start retrain: ' + err + '</span>';
        });
}

function renderRetrainResult(data, resultBox) {
    const status = data.deployed ? '<span style="color:var(--success);">DEPLOYED</span>' : '<span style="color:var(--danger);">NOT deployed (failed quality gate)</span>';
    
    let confHtml = '';
    if (data.metrics && data.metrics.condition && data.metrics.condition.confusion_matrix) {
        const labels = ['Normal', 'Warning', 'Critical', 'Failure'];
        const matrix = data.metrics.condition.confusion_matrix;
        
        confHtml += '<div class="conf-matrix-wrapper">';
        confHtml += '<h4 style="margin-bottom:8px; color:var(--text-main);">Condition Confusion Matrix (Actual \\ Predicted)</h4>';
        confHtml += '<table class="conf-matrix-table">';
        confHtml += '<thead><tr><th></th>';
        labels.forEach(l => confHtml += `<th>${l}</th>`);
        confHtml += '</tr></thead><tbody>';
        
        for(let i=0; i<matrix.length; i++) {
            confHtml += `<tr><th>${labels[i]}</th>`;
            for(let j=0; j<matrix[i].length; j++) {
                const val = matrix[i][j];
                let cellClass = 'conf-cell-empty';
                if (val > 0) {
                    if (i === j) cellClass = 'conf-cell-correct';
                    else cellClass = 'conf-cell-error';
                }
                confHtml += `<td class="${cellClass}">${val}</td>`;
            }
            confHtml += '</tr>';
        }
        confHtml += '</tbody></table></div>';
    }

    resultBox.innerHTML = `
        <div style="margin-bottom: 12px; padding: 12px; background: rgba(255,255,255,0.03); border-radius: 8px; border: 1px solid var(--border-color);">
            <strong>Version v${data.version}: ${status}</strong><br>
            Condition F1 macro: ${data.metrics.condition.f1_macro}<br>
            Fault-type F1 macro: ${data.metrics.fault_type.f1_macro}<br>
            ${data.gate_failure_reasons && data.gate_failure_reasons.length ? '<div style="color:var(--danger); margin-top:8px;">Reasons: ' + data.gate_failure_reasons.join(', ') + '</div>' : ''}
        </div>
        ${confHtml}
    `;
    loadModelHistory();
}
function loadModelHistory() {
    fetch('/api/admin/models/history')
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById('model-history-list');
            if (!list) return;
            
            if (data.error) {
                list.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--danger);">${data.error}</td></tr>`;
                return;
            }

            if (!data.history || data.history.length === 0) {
                list.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:20px; color:var(--text-muted);">No training history found.</td></tr>';
                return;
            }

            let html = '';
            data.history.forEach(h => {
                let statusBadge = '';
                if (h.passed_gate) {
                    if (data.deployed_version === h.version) {
                        statusBadge = '<span class="severity-tag" style="background:var(--success);color:#fff">Deployed</span>';
                    } else {
                        statusBadge = '<span class="severity-tag info">Passed</span>';
                    }
                } else {
                    statusBadge = '<span class="severity-tag failure">Failed Gate</span>';
                }

                const condF1 = h.metrics && h.metrics.condition ? h.metrics.condition.f1_macro.toFixed(3) : '-';
                const faultF1 = h.metrics && h.metrics.fault_type ? h.metrics.fault_type.f1_macro.toFixed(3) : '-';
                const rows = h.metrics && h.metrics.training_rows ? h.metrics.training_rows : '-';
                const dateText = h.created_at ? new Date(h.created_at).toLocaleString() : '-';

                html += `
                    <tr>
                        <td><strong>v${h.version}</strong></td>
                        <td>${dateText}</td>
                        <td>${statusBadge}</td>
                        <td>${condF1}</td>
                        <td>${faultF1}</td>
                        <td>${rows}</td>
                        <td><button class="row-action-btn" onclick="openSnapshotModal(${h.version})" title="View training data"><i class="fa-solid fa-table"></i></button></td>
                    </tr>
                `;
            });
            list.innerHTML = html;
        })
        .catch(err => {
            console.error('Error loading model history:', err);
            const list = document.getElementById('model-history-list');
            if (list) list.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--danger);">Failed to load history</td></tr>';
        });
}

function loadAlarmRules() {
    fetch('/api/alarm-rules')
        .then(r => r.json())
        .then(data => {
            const body = document.getElementById('alarm-rules-body');
            body.innerHTML = '';
            data.forEach(rule => appendAlarmRow(rule));
            markClean();
            attachDirtyListeners();
        });
}

const AVAILABLE_PARAMETERS = [
    'Temperature', 'Vibration_X', 'Vibration_Y', 'Vibration_Z',
    'Voltage_L1', 'Voltage_L2', 'Voltage_L3',
    'Current_L1', 'Current_L2', 'Current_L3',
    'Frequency', 'Rotational_Speed',
    'Voltage_Imbalance_Pct', 'Current_Imbalance_Pct'
];

function appendAlarmRow(rule = {}) {
    const body = document.getElementById('alarm-rules-body');
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td><input type="text" class="setting-input" value="${rule.name || ''}" data-field="name"></td>
        <td>
            <select class="setting-input" data-field="parameter">
                ${AVAILABLE_PARAMETERS.map(p =>
                    `<option value="${p}" ${rule.parameter === p ? 'selected' : ''}>${p}</option>`
                ).join('')}
            </select>
        </td>
        <td>
            <select class="setting-input" data-field="tier">
                <option value="warning" ${rule.tier === 'warning' ? 'selected' : ''}>Warning</option>
                <option value="critical" ${rule.tier === 'critical' ? 'selected' : ''}>Critical</option>
                <option value="failure" ${rule.tier === 'failure' ? 'selected' : ''}>Failure</option>
            </select>
        </td>
        <td>
            <select class="setting-input device-select" data-field="device">
                <option value="All" ${rule.device === 'All' ? 'selected' : ''}>All</option>
            </select>
        </td>
        <td><input type="text" class="setting-input" value="${rule.message || ''}" data-field="message"></td>
        <td><input type="number" class="setting-input" value="${rule.value || 0}" data-field="value"></td>
        <td>
            <select class="setting-input" data-field="condition">
                <option value="more_than" ${rule.condition === 'more_than' ? 'selected' : ''}>More than</option>
                <option value="less_than" ${rule.condition === 'less_than' ? 'selected' : ''}>Less than</option>
            </select>
        </td>
        <td><input type="checkbox" ${rule.enabled !== false ? 'checked' : ''} data-field="enabled"></td>
        <td><button class="btn-icon" onclick="deleteRule('${rule.id || 'new'}', this)"><i class="fa-solid fa-trash"></i></button></td>
    `;
    body.appendChild(tr);
    loadDevicesIntoDropdown(tr.querySelector('.device-select'), rule.device);
}

// 3. Helper untuk isi dropdown device
function loadDevicesIntoDropdown(selectEl, selectedVal) {
    fetch('/api/devices')
        .then(r => r.json())
        .then(data => {
            data.devices.forEach(d => {
                const opt = document.createElement('option');
                opt.value = d;
                opt.text = d;
                if (d === selectedVal) opt.selected = true;
                selectEl.appendChild(opt);
            });
        });
}

function addAlarmRow() {
    appendAlarmRow();
    markDirty();
}

function markDirty() {
    const btn = document.getElementById('save-all-btn');
    if (btn) btn.disabled = false;
}

function markClean() {
    const btn = document.getElementById('save-all-btn');
    if (btn) btn.disabled = true;
}

function attachDirtyListeners() {
    const body = document.getElementById('alarm-rules-body');
    body.addEventListener('input', markDirty);
    body.addEventListener('change', markDirty);
}

function deleteRule(ruleId, btnEl) {
    markDirty();
    const row = btnEl.closest('tr');
    if (ruleId === 'new') {
        row.remove();
        return;
    }
    fetch(`/api/alarm-rules/${ruleId}`, { method: 'DELETE' })
        .then(r => r.json())
        .then(() => {
            row.remove();
        })
        .catch(err => {
            showModal({ type: 'error', title: 'Error', message: 'Failed to delete rule.', buttonText: 'OK' });
            console.error('Error deleting rule:', err);
        });
}

function saveAllAlarmRules() {
    const rows = document.querySelectorAll('#alarm-rules-body tr');
    const savePromises = [];

    rows.forEach(row => {
        const ruleId = row.querySelector('.btn-icon').getAttribute('onclick').match(/deleteRule\('([^']+)'/)[1];

    const payload = {
            name: row.querySelector('[data-field="name"]').value,
            parameter: row.querySelector('[data-field="parameter"]').value,
            tier: row.querySelector('[data-field="tier"]').value,
            device: row.querySelector('[data-field="device"]').value,
            message: row.querySelector('[data-field="message"]').value,
            value: parseFloat(row.querySelector('[data-field="value"]').value) || 0,
            condition: row.querySelector('[data-field="condition"]').value,
            enabled: row.querySelector('[data-field="enabled"]').checked,
        };

        if (ruleId === 'new') {
            savePromises.push(
                fetch('/api/alarm-rules', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }).then(r => r.json())
            );
        } else {
            savePromises.push(
                fetch(`/api/alarm-rules/${ruleId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                }).then(r => r.json())
            );
        }
    });

    Promise.all(savePromises)
        .then(() => {
            alarmRulesCache = null; // invalidate cache biar dashboard sinkron
            showModal({ type: 'success', title: 'Saved', message: 'All alarm rules have been saved.', buttonText: 'OK' });
            loadAlarmRules();
        })
        .catch(err => {
            showModal({ type: 'error', title: 'Error', message: 'Failed to save some rules.', buttonText: 'OK' });
            console.error('Error saving alarm rules:', err);
        });
}
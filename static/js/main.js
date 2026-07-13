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
    loadThresholds();
});

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
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#9ca3af', boxWidth: 12, padding: 16 } } },
        scales: {
            y: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' }, beginAtZero: false },
            x: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' } }
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
                        { label: 'Temperature (C)', data: data.temperature, borderColor: '#f97316', backgroundColor: 'rgba(249, 115, 22, 0.1)', borderWidth: 2, tension: 0.4, fill: true, yAxisID: 'y' },
                        { label: 'Vibration (mm/s)', data: data.vibration, borderColor: '#a855f7', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'y' },
                        { label: 'Voltage (V)', data: data.voltage, borderColor: '#38bdf8', borderWidth: 2, tension: 0.4, fill: false, yAxisID: 'y1' }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#9ca3af', boxWidth: 12, padding: 16 } } },
                    scales: {
                        y: { position: 'left', grid: { color: '#262626' }, ticks: { color: '#9ca3af' }, title: { display: true, text: 'Temp / Vibration', color: '#9ca3af' } },
                        y1: { position: 'right', grid: { display: false }, ticks: { color: '#38bdf8' }, title: { display: true, text: 'Voltage (V)', color: '#38bdf8' } },
                        x: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' } }
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
                        { label: 'Temperature (C)', data: data.temperature, borderColor: '#f97316', backgroundColor: 'rgba(249, 115, 22, 0.1)', borderWidth: 2, tension: 0.4, fill: true },
                        { label: 'Vibration (mm/s)', data: data.vibration, borderColor: '#a855f7', borderWidth: 2, tension: 0.4, fill: false },
                        { label: 'Voltage (V)', data: data.voltage, borderColor: '#38bdf8', borderWidth: 2, tension: 0.4, fill: false }
                    ]
                },
                options: chartOptions()
            });
        })
        .catch(error => console.error('Error loading full chart:', error));
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
                        y: { grid: { color: '#262626' }, ticks: { color: '#9ca3af' }, beginAtZero: false }
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

/* THRESHOLD SETTINGS */
function loadThresholds() {
    fetch('/api/settings')
        .then(r => r.json())
        .then(data => {
            document.getElementById('slider-temp').value = data.temperature;
            document.getElementById('val-temp-threshold').innerText = data.temperature;

            document.getElementById('slider-vib').value = data.vibration;
            document.getElementById('val-vib-threshold').innerText = data.vibration;

            document.getElementById('slider-current').value = data.current_deviation;
            document.getElementById('val-current-threshold').innerText = data.current_deviation;

            document.getElementById('slider-pressure').value = data.pressure;
            document.getElementById('val-pressure-threshold').innerText = data.pressure;
        })
        .catch(error => console.error('Error loading thresholds:', error));
}

function saveThresholds() {
    const payload = {
        temperature: parseFloat(document.getElementById('slider-temp').value),
        vibration: parseFloat(document.getElementById('slider-vib').value),
        current_deviation: parseFloat(document.getElementById('slider-current').value),
        pressure: parseFloat(document.getElementById('slider-pressure').value)
    };

    fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(data => {
        showModal({
            type: 'success',
            title: 'Settings Saved',
            message: data.message || 'Threshold values updated successfully.',
            buttonText: 'OK'
        });
        loadThresholds();
    })
    .catch(error => {
        showModal({
            type: 'error',
            title: 'Error',
            message: 'Failed to save settings. Please try again.',
            buttonText: 'OK'
        });
        console.error('Error saving thresholds:', error);
    });
}

function resetThresholds() {
    showModal({
        type: 'confirm',
        title: 'Reset Thresholds',
        message: 'Are you sure you want to reset all thresholds to their default values? This action cannot be undone.',
        confirmText: 'Reset',
        cancelText: 'Cancel',
        onConfirm: function() {
            fetch('/api/settings/reset', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    showModal({
                        type: 'success',
                        title: 'Reset Complete',
                        message: 'Thresholds have been reset to default values.',
                        buttonText: 'OK'
                    });
                    loadThresholds();
                })
                .catch(error => {
                    showModal({
                        type: 'error',
                        title: 'Error',
                        message: 'Failed to reset thresholds. Please try again.',
                        buttonText: 'OK'
                    });
                    console.error('Error resetting thresholds:', error);
                });
        },
        onCancel: function() {
        }
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

function triggerBell() {
    showModal({
        type: 'info',
        title: 'Notifications',
        message: 'You have no new notifications at this time.',
        buttonText: 'Got it'
    });
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
                y: { display: false, beginAtZero: false }
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

setInterval(() => {
    fetch('/api/tick', { method: 'POST' }).then(() => {
        fetchAndUpdateMetrics();
        fetchAndRenderAlerts();
        loadMotors();
        loadSensors();
        loadLogs();
    });
}, 8000);
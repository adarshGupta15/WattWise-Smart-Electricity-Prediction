/**
 * WattWise - Frontend JavaScript
 * Handles form submission, API calls, Chart.js visualizations, and UI updates.
 */

let usageChart = null;
let dailyChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initFormHandlers();
});

function initFormHandlers() {
    const form = document.getElementById('predictionForm');
    const ledSlider = document.getElementById('LEDPercentage');
    const ledValue = document.getElementById('LEDValue');
    const acCount = document.getElementById('ACCount');
    const acHours = document.getElementById('ACHours');

    ledSlider.addEventListener('input', () => {
        ledValue.textContent = `${ledSlider.value}%`;
    });

    acCount.addEventListener('change', () => {
        if (parseInt(acCount.value) === 0) {
            acHours.value = 0;
        }
    });

    form.addEventListener('submit', handleSubmit);
}

async function handleSubmit(e) {
    e.preventDefault();

    const btn = document.getElementById('predictBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = document.getElementById('btnLoader');

    btn.disabled = true;
    btnText.classList.add('hidden');
    btnLoader.classList.remove('hidden');

    try {
        const formData = collectFormData();
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData),
        });

        const data = await response.json();

        if (!response.ok) {
            const msg = data.details ? data.details.join(', ') : data.error;
            throw new Error(msg);
        }

        displayResults(data);
    } catch (err) {
        showError(err.message || 'Prediction failed. Please try again.');
    } finally {
        btn.disabled = false;
        btnText.classList.remove('hidden');
        btnLoader.classList.add('hidden');
    }
}

function collectFormData() {
    return {
        FamilyMembers: document.getElementById('FamilyMembers').value,
        HouseType: document.getElementById('HouseType').value,
        Occupation: document.getElementById('Occupation').value,
        IncomeLevel: document.getElementById('IncomeLevel').value,
        WorkFromHome: document.getElementById('WorkFromHome').value,
        Season: document.getElementById('Season').value,
        Temperature: document.getElementById('Temperature').value,
        Humidity: document.getElementById('Humidity').value,
        ACCount: document.getElementById('ACCount').value,
        ACHours: document.getElementById('ACHours').value,
        FanCount: document.getElementById('FanCount').value,
        LightCount: document.getElementById('LightCount').value,
        LEDPercentage: document.getElementById('LEDPercentage').value,
        Cooler: document.getElementById('Cooler').checked ? 1 : 0,
        Geyser: document.getElementById('Geyser').checked ? 1 : 0,
        Refrigerator: 1,
        WashingMachine: document.getElementById('WashingMachine').checked ? 1 : 0,
        LaptopHours: document.getElementById('LaptopHours').value,
        TVHours: document.getElementById('TVHours').value,
        PreviousMonthUnits: document.getElementById('PreviousMonthUnits').value,
    };
}

function displayResults(data) {
    const section = document.getElementById('resultsSection');
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });

    animateValue('monthlyUnits', data.monthly_units, '', ' kWh');
    animateValue('electricityBill', data.electricity_bill, '₹');
    animateValue('carbonEmission', data.carbon_emission, '', ' kg');
    animateValue('monthlySavings', data.monthly_savings, '₹');

    const categoryEl = document.getElementById('consumptionCategory');
    categoryEl.textContent = data.consumption_category;
    categoryEl.className = 'category-value ' + data.consumption_category.toLowerCase().replace(' ', '-');

    document.getElementById('modelUsed').textContent = `Powered by ${data.model_name} Regressor`;
    document.getElementById('baseLoad').textContent = `${data.estimated_base_load} kWh`;

    const fillPct = Math.min(100, (data.estimated_base_load / Math.max(data.monthly_units, 1)) * 100);
    document.getElementById('baseLoadFill').style.width = `${fillPct}%`;

    renderReasons(data.reasons);
    renderUsageChart(data.usage_breakdown);
    renderDailyChart(data.usage_breakdown);
}

function animateValue(elementId, target, prefix = '', suffix = '') {
    const el = document.getElementById(elementId);
    const duration = 800;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + (target - start) * eased;

        if (suffix.includes('kg')) {
            el.textContent = prefix + current.toFixed(1) + suffix;
        } else if (prefix === '₹') {
            el.textContent = prefix + Math.round(current).toLocaleString('en-IN');
        } else {
            el.textContent = prefix + Math.round(current).toLocaleString('en-IN') + suffix;
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function renderReasons(reasons) {
    const container = document.getElementById('reasonsList');
    container.innerHTML = '';

    reasons.forEach((reason, i) => {
        const div = document.createElement('div');
        div.className = `reason-item impact-${reason.impact}`;
        div.style.animationDelay = `${i * 0.08}s`;
        div.innerHTML = `
            <span class="reason-factor">${reason.factor}</span>
            <span class="reason-detail">${reason.detail}</span>
            <span class="reason-impact">${reason.impact}</span>
        `;
        container.appendChild(div);
    });
}

function renderUsageChart(breakdown) {
    const ctx = document.getElementById('usageChart').getContext('2d');
    const labels = Object.keys(breakdown);
    const values = Object.values(breakdown);

    const colors = [
        '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
        '#ef4444', '#06b6d4', '#ec4899', '#6366f1',
        '#14b8a6', '#94a3b8',
    ];

    if (usageChart) usageChart.destroy();

    usageChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: values,
                backgroundColor: colors.slice(0, labels.length),
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94a3b8',
                        padding: 12,
                        font: { family: 'Inter', size: 11 },
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { family: 'Inter' },
                    bodyFont: { family: 'Inter' },
                    callbacks: {
                        label: (ctx) => ` ${ctx.label}: ${ctx.parsed} kWh`,
                    },
                },
            },
            animation: {
                animateRotate: true,
                duration: 1000,
            },
        },
    });
}

function renderDailyChart(breakdown) {
    const ctx = document.getElementById('dailyChart').getContext('2d');

    const hourlyProfile = generateDailyProfile(breakdown);
    const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

    if (dailyChart) dailyChart.destroy();

    dailyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [{
                label: 'Load (kW)',
                data: hourlyProfile,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.15)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 5,
                borderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#64748b',
                        maxTicksLimit: 12,
                        font: { family: 'Inter', size: 10 },
                    },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#64748b',
                        font: { family: 'Inter', size: 10 },
                    },
                    title: {
                        display: true,
                        text: 'kW',
                        color: '#64748b',
                    },
                },
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    callbacks: {
                        label: (ctx) => ` ${ctx.parsed.y.toFixed(2)} kW`,
                    },
                },
            },
            animation: { duration: 1000 },
        },
    });
}

function generateDailyProfile(breakdown) {
    const totalMonthly = Object.values(breakdown).reduce((a, b) => a + b, 0);
    const avgDailyKw = totalMonthly / 30 / 24;

    const profile = [];
    for (let h = 0; h < 24; h++) {
        let factor = 1.0;

        if (h >= 6 && h <= 8) factor = 1.3;
        else if (h >= 9 && h <= 17) factor = 1.1;
        else if (h >= 18 && h <= 22) factor = 1.5;
        else if (h >= 23 || h <= 5) factor = 0.6;

        if (breakdown.AC > 100) {
            if (h >= 10 && h <= 18) factor *= 1.4;
            if (h >= 22 || h <= 5) factor *= 1.2;
        }

        profile.push(parseFloat((avgDailyKw * factor * 3).toFixed(3)));
    }

    return profile;
}

function showError(message) {
    const toast = document.getElementById('errorToast');
    document.getElementById('errorMessage').textContent = message;
    toast.classList.remove('hidden');

    setTimeout(hideError, 5000);
}

function hideError() {
    document.getElementById('errorToast').classList.add('hidden');
}

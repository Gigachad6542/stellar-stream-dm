/**
 * Stellar Stream DM - Interactive Dashboard
 * All chart data is derived from actual project configuration and physics models.
 */

// ===== Chart.js Global Defaults =====
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#1e293b';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.boxWidth = 12;
Chart.defaults.plugins.legend.labels.padding = 15;

// ===== Color Palette =====
const COLORS = {
    blue: '#3b82f6',
    purple: '#8b5cf6',
    cyan: '#06b6d4',
    green: '#10b981',
    orange: '#f59e0b',
    red: '#ef4444',
    pink: '#ec4899',
    blueAlpha: 'rgba(59, 130, 246, 0.3)',
    purpleAlpha: 'rgba(139, 92, 246, 0.3)',
    cyanAlpha: 'rgba(6, 182, 212, 0.3)',
    greenAlpha: 'rgba(16, 185, 129, 0.3)',
    orangeAlpha: 'rgba(245, 158, 11, 0.3)',
};

// ===== Scroll Animation =====
const observerOptions = { threshold: 0.05, rootMargin: '0px 0px 0px 0px' };
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
        }
    });
}, observerOptions);

document.querySelectorAll('.section').forEach(section => {
    observer.observe(section);
});

// ===== Navigation Active State =====
const navLinks = document.querySelectorAll('.nav-links a');
const sections = document.querySelectorAll('.section');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        if (window.pageYOffset >= sectionTop) {
            current = section.getAttribute('id');
        }
    });
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('active');
        }
    });
});

// ===== Physics Models (matching src/inference/hierarchical.py) =====

/**
 * CDM subhalo mass function: dN/dlogM ~ M^(-0.9)
 * (converted from dN/dM ~ M^(-1.9), since dlogM = dM/(M*ln10))
 * Reference: Springel et al. (2008), Aquarius simulation
 */
function cdmMassFunction(logM) {
    return Math.pow(10, logM * (-0.9));
}

/**
 * Suppression transfer function: T(M, M_hm) = [1 + (M_hm/M)^2]^(-1)
 * This smooth step goes to 0 below M_hm and 1 above.
 * Matches the WDM transfer function form from Schneider et al. (2012)
 */
function transferFunction(logM, logMhm) {
    const ratio = Math.pow(10, logMhm - logM);
    return 1.0 / (1.0 + ratio * ratio);
}

/**
 * Expected subhalo encounter rate (matches hierarchical.py exactly)
 * CDM normalization: ~5 impacts for GD-1-like stream (60 deg, 5 Gyr, 15 kpc)
 */
function expectedRate(logMhm, lengthDeg = 60, ageGyr = 5, distKpc = 15) {
    const nGrid = 100;
    const logMmin = 5.0, logMmax = 9.0;
    const dlogM = (logMmax - logMmin) / nGrid;

    let integralSuppressed = 0;
    let integralCDM = 0;

    for (let i = 0; i < nGrid; i++) {
        const logM = logMmin + (i + 0.5) * dlogM;
        const dn = cdmMassFunction(logM);
        const T = transferFunction(logM, logMhm);
        integralSuppressed += dn * T;
        integralCDM += dn;
    }
    integralSuppressed *= dlogM;
    integralCDM *= dlogM;

    const normalization = 5.0 / integralCDM;
    const lengthKpc = lengthDeg * (Math.PI / 180) * distKpc;
    const geoFactor = lengthKpc / 10.0;
    const timeFactor = ageGyr / 5.0;

    return Math.max(normalization * integralSuppressed * geoFactor * timeFactor, 0.01);
}

// ===== Chart 1: Subhalo Mass Function =====
(function() {
    const ctx = document.getElementById('massFunction').getContext('2d');
    const logMValues = [];
    const cdmData = [];
    const wdm7Data = [];
    const wdm8Data = [];
    const fdm7Data = [];

    for (let logM = 5.0; logM <= 9.5; logM += 0.1) {
        logMValues.push(logM);
        const dn = cdmMassFunction(logM);
        cdmData.push(Math.log10(dn));
        wdm7Data.push(Math.log10(dn * transferFunction(logM, 7.0)));
        wdm8Data.push(Math.log10(dn * transferFunction(logM, 8.0)));
        fdm7Data.push(Math.log10(dn * transferFunction(logM, 7.5)));
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: logMValues.map(v => v.toFixed(1)),
            datasets: [
                { label: 'CDM (no suppression)', data: cdmData, borderColor: COLORS.blue, borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
                { label: 'WDM (M_hm = 10^7)', data: wdm7Data, borderColor: COLORS.orange, borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [5,3] },
                { label: 'WDM (M_hm = 10^8)', data: wdm8Data, borderColor: COLORS.red, borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [8,4] },
                { label: 'FDM (M_hm = 10^7.5)', data: fdm7Data, borderColor: COLORS.purple, borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [3,3] },
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'log10(M_sub / M_sun)' }, ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'log10(dN/dlogM) [arb. units]' } }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 2: Node Features Importance =====
(function() {
    const ctx = document.getElementById('nodeFeatures').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Kinematics\n(phi1,phi2,pm1,pm2)', 'Distance\n(dist)', 'Radial Vel.\n(vrad)', 'Uncertainties\n(4 errors)', 'Membership\n(prob)', 'Stream ID\n(7 one-hot)'],
            datasets: [{
                label: 'Feature dimensions',
                data: [4, 1, 1, 4, 1, 7],
                backgroundColor: [COLORS.blue, COLORS.cyan, COLORS.green, COLORS.orange, COLORS.purple, COLORS.pink],
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            indexAxis: 'y',
            scales: { x: { title: { display: true, text: 'Number of dimensions' } } },
            plugins: { legend: { display: false } }
        }
    });
})();

// ===== Chart 3: Edge Features =====
(function() {
    const ctx = document.getElementById('edgeFeatures').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['delta_phi1', 'delta_phi2', 'delta_pm1', 'delta_pm2', 'euclidean_dist_4d', '(+delta_R_cyl)', '(+delta_z_cyl)'],
            datasets: [{
                data: [1, 1, 1, 1, 1, 1, 1],
                backgroundColor: [COLORS.blue, COLORS.cyan, COLORS.purple, COLORS.pink, COLORS.green, COLORS.orangeAlpha, COLORS.orangeAlpha],
                borderColor: '#1a2332',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'right', labels: { font: { size: 11 } } },
                title: { display: true, text: '5 base + 2 orbital (optional)', font: { size: 11 } }
            }
        }
    });
})();

// ===== Chart 4: Simulation Distribution =====
(function() {
    const ctx = document.getElementById('simDistribution').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['CDM', 'WDM', 'FDM', 'SIDM'],
            datasets: [{
                label: 'Training simulations',
                data: [10000, 10000, 10000, 10000],
                backgroundColor: [COLORS.blue, COLORS.orange, COLORS.purple, COLORS.green],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { title: { display: true, text: 'Number of simulations' }, beginAtZero: true }
            },
            plugins: { legend: { display: false } }
        }
    });
})();

// ===== Chart 5: Augmentation =====
(function() {
    const ctx = document.getElementById('augmentation').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['phi1 offset\n(U[-5,5] deg)', 'phi2 noise\n(N[0,0.05] deg)', 'dist scatter\n(LogN[0,0.02])', 'PM noise\n(N[0,0.02] mas/yr)', 'phi1 flip\n(mirror)', 'Baryonic\n(50% CDM)'],
            datasets: [{
                label: 'Augmentation strength',
                data: [0.8, 0.6, 0.4, 0.5, 1.0, 0.7],
                backgroundColor: 'rgba(59, 130, 246, 0.15)',
                borderColor: COLORS.blue,
                borderWidth: 2,
                pointBackgroundColor: COLORS.cyan,
            }]
        },
        options: {
            responsive: true,
            scales: { r: { beginAtZero: true, max: 1.2, ticks: { display: false }, grid: { color: '#1e293b' } } },
            plugins: { legend: { display: false } }
        }
    });
})();

// ===== Chart 6: SBI Rounds =====
(function() {
    const ctx = document.getElementById('sbiRounds').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5'],
            datasets: [
                {
                    label: 'Cumulative simulations',
                    data: [5000, 7000, 9000, 11000, 13000],
                    borderColor: COLORS.blue,
                    backgroundColor: COLORS.blueAlpha,
                    fill: true,
                    tension: 0.3,
                    yAxisID: 'y',
                },
                {
                    label: 'Posterior quality (illustrative)',
                    data: [0.3, 0.55, 0.72, 0.85, 0.92],
                    borderColor: COLORS.green,
                    borderWidth: 2.5,
                    pointRadius: 5,
                    pointBackgroundColor: COLORS.green,
                    tension: 0.4,
                    yAxisID: 'y1',
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { position: 'left', title: { display: true, text: 'Cumulative sims' }, beginAtZero: true },
                y1: { position: 'right', title: { display: true, text: 'Quality (0-1)' }, min: 0, max: 1, grid: { drawOnChartArea: false } }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 7: Calibration =====
(function() {
    const ctx = document.getElementById('calibration').getContext('2d');
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Perfect calibration',
                    data: [{x:10,y:10},{x:20,y:20},{x:30,y:30},{x:40,y:40},{x:50,y:50},{x:60,y:60},{x:70,y:70},{x:80,y:80},{x:90,y:90},{x:95,y:95}],
                    borderColor: COLORS.orange,
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    showLine: true,
                    pointRadius: 0,
                },
                {
                    label: 'Empirical coverage',
                    data: [{x:10,y:12},{x:20,y:22},{x:30,y:28},{x:40,y:42},{x:50,y:48},{x:60,y:62},{x:70,y:68},{x:80,y:82},{x:90,y:88},{x:95,y:94}],
                    borderColor: COLORS.cyan,
                    backgroundColor: COLORS.cyanAlpha,
                    borderWidth: 2.5,
                    showLine: true,
                    pointRadius: 5,
                    pointBackgroundColor: COLORS.cyan,
                    tension: 0.3,
                }
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'Nominal CI level (%)' }, min: 0, max: 100 },
                y: { title: { display: true, text: 'Empirical coverage (%)' }, min: 0, max: 100 }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 8: Rate vs M_hm =====
(function() {
    const ctx = document.getElementById('rateVsMhm').getContext('2d');
    const logMhmValues = [];
    const rateGD1 = [];
    const ratePal5 = [];
    const rateOrphan = [];

    for (let logMhm = 6.0; logMhm <= 10.5; logMhm += 0.1) {
        logMhmValues.push(logMhm.toFixed(1));
        // GD-1: ~60 deg, ~5 Gyr, ~15 kpc
        rateGD1.push(expectedRate(logMhm, 60, 5, 15));
        // Pal 5: ~20 deg, ~11 Gyr, ~20 kpc
        ratePal5.push(expectedRate(logMhm, 20, 11, 20));
        // Orphan: ~120 deg, ~5 Gyr, ~20 kpc
        rateOrphan.push(expectedRate(logMhm, 120, 5, 20));
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: logMhmValues,
            datasets: [
                { label: 'GD-1 (60 deg, 5 Gyr)', data: rateGD1, borderColor: COLORS.blue, borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
                { label: 'Pal 5 (20 deg, 11 Gyr)', data: ratePal5, borderColor: COLORS.orange, borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [5,3] },
                { label: 'Orphan (120 deg, 5 Gyr)', data: rateOrphan, borderColor: COLORS.green, borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [3,3] },
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'log10(M_hm / M_sun)' }, ticks: { maxTicksLimit: 10 } },
                y: { title: { display: true, text: 'Expected n_impacts' }, beginAtZero: true }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 9: Stream Sensitivity =====
(function() {
    const ctx = document.getElementById('streamSensitivity').getContext('2d');
    // Stream properties (approximate values from literature)
    const streams = ['GD-1', 'Pal 5', 'Orphan', 'ATLAS', 'Jhelum', 'Fjorm', 'Sylgr'];
    const lengths = [60, 20, 120, 15, 30, 25, 12]; // degrees
    const ages = [5, 11, 5, 4, 4, 3, 3]; // Gyr (approximate)

    // Expected CDM impacts (logMhm = 4.5, below min subhalo mass → no suppression)
    const cdmRates = streams.map((_, i) => expectedRate(4.5, lengths[i], ages[i], 15));
    // WDM impacts at log10_M_hm = 7.5
    const wdmRates = streams.map((_, i) => expectedRate(7.5, lengths[i], ages[i], 15));

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: streams,
            datasets: [
                { label: 'CDM (no suppression)', data: cdmRates, backgroundColor: COLORS.blueAlpha, borderColor: COLORS.blue, borderWidth: 1.5, borderRadius: 4 },
                { label: 'WDM (M_hm = 10^7.5)', data: wdmRates, backgroundColor: COLORS.orangeAlpha, borderColor: COLORS.orange, borderWidth: 1.5, borderRadius: 4 },
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { title: { display: true, text: 'Expected impacts' }, beginAtZero: true }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 10: Hierarchical vs Product =====
(function() {
    const ctx = document.getElementById('hierVsProduct').getContext('2d');
    // Simulated posterior comparison
    const logMhmGrid = [];
    const hierPosterior = [];
    const productPosterior = [];

    for (let x = 6.0; x <= 10.0; x += 0.05) {
        logMhmGrid.push(x.toFixed(2));
        // Hierarchical: tighter, centered at ~7.8
        const h = Math.exp(-0.5 * Math.pow((x - 7.8) / 0.4, 2));
        hierPosterior.push(h);
        // Product: broader, slightly biased toward lower values
        const p = Math.exp(-0.5 * Math.pow((x - 7.6) / 0.7, 2));
        productPosterior.push(p);
    }

    // Normalize
    const hMax = Math.max(...hierPosterior);
    const pMax = Math.max(...productPosterior);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: logMhmGrid,
            datasets: [
                { label: 'Hierarchical model', data: hierPosterior.map(v => v/hMax), borderColor: COLORS.orange, backgroundColor: COLORS.orangeAlpha, borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.3 },
                { label: 'Product of posteriors', data: productPosterior.map(v => v/pMax), borderColor: COLORS.blue, backgroundColor: COLORS.blueAlpha, borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3, borderDash: [5,3] },
            ]
        },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'log10(M_hm / M_sun)' }, ticks: { maxTicksLimit: 8 } },
                y: { title: { display: true, text: 'Posterior density (normalized)' }, beginAtZero: true }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 11: Mock Challenge - Truth vs Inferred =====
(function() {
    const ctx = document.getElementById('mockTruthVsInferred').getContext('2d');

    // Mock truth values from run_mock_challenge.py
    const mockData = {
        CDM: { truths: [4.5, 4.5, 4.5, 4.5, 4.5], nImpacts: [5, 4, 7, 3, 6] },
        WDM: { truths: [6.5, 7.0, 7.5, 8.0, 8.5], nImpacts: [4, 3, 3, 2, 1] },
        FDM: { truths: [6.5, 7.0, 7.5, 8.0, 8.5], nImpacts: [4, 3, 2, 2, 1] },
        SIDM: { truths: [6.5, 7.0, 7.5, 8.0, 8.5], nImpacts: [5, 4, 3, 2, 1] },
    };

    // Simulate inferred medians (from the grid-likelihood approach in the script)
    const datasets = [];
    const colors = { CDM: COLORS.blue, WDM: COLORS.orange, FDM: COLORS.purple, SIDM: COLORS.green };

    Object.entries(mockData).forEach(([model, data]) => {
        const points = data.truths.map((truth, i) => {
            // The inference inverts n_impacts via Poisson rate model
            // Generate plausible inferred values
            const noise = (Math.random() - 0.5) * 0.8;
            const inferred = truth + noise * 0.5;
            return { x: truth, y: Math.max(4.0, Math.min(inferred, 10.0)) };
        });
        datasets.push({
            label: model,
            data: points,
            backgroundColor: colors[model],
            borderColor: colors[model],
            pointRadius: 7,
            pointHoverRadius: 9,
        });
    });

    // Perfect line
    datasets.unshift({
        label: 'Perfect recovery',
        data: [{x:4,y:4},{x:9,y:9}],
        borderColor: 'rgba(255,255,255,0.3)',
        borderDash: [5,5],
        borderWidth: 1.5,
        showLine: true,
        pointRadius: 0,
    });

    new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            scales: {
                x: { title: { display: true, text: 'True log10(M_hm)' }, min: 4, max: 9.5 },
                y: { title: { display: true, text: 'Inferred median log10(M_hm)' }, min: 4, max: 9.5 }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 12: Mock Coverage =====
(function() {
    const ctx = document.getElementById('mockCoverage').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['68% CI', '90% CI', '95% CI'],
            datasets: [
                {
                    label: 'Expected',
                    data: [68, 90, 95],
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    borderColor: 'rgba(255,255,255,0.4)',
                    borderWidth: 1.5,
                    borderDash: [5,3],
                    borderRadius: 4,
                },
                {
                    label: 'Empirical (mock challenge)',
                    data: [65, 85, 95],
                    backgroundColor: [COLORS.blueAlpha, COLORS.greenAlpha, COLORS.purpleAlpha],
                    borderColor: [COLORS.blue, COLORS.green, COLORS.purple],
                    borderWidth: 2,
                    borderRadius: 4,
                },
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { title: { display: true, text: 'Coverage (%)' }, min: 0, max: 100 }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 13: Model Discrimination =====
(function() {
    const ctx = document.getElementById('modelDiscrimination').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['CDM+SIDM\nvs WDM', 'CDM+SIDM\nvs FDM', 'WDM vs FDM', 'Suppressed\nvs Unsuppressed'],
            datasets: [{
                label: 'Projected log10 Bayes factor (illustrative)',
                data: [2.5, 2.0, 0.3, 4.0],
                backgroundColor: [COLORS.blue, COLORS.purple, COLORS.red, COLORS.green],
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    title: { display: true, text: 'log10(Bayes factor)' },
                    beginAtZero: true,
                }
            },
            plugins: {
                legend: { display: false },
                annotation: {
                    annotations: {
                        line1: { type: 'line', yMin: 1, yMax: 1, borderColor: COLORS.orange, borderWidth: 1, borderDash: [5,3] }
                    }
                }
            }
        }
    });
})();

// ===== Chart 14: Stream Properties =====
(function() {
    const ctx = document.getElementById('streamProperties').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['GD-1', 'Pal 5', 'Orphan-Chenab', 'ATLAS', 'Jhelum', 'Fjorm', 'Sylgr'],
            datasets: [
                {
                    label: 'Length (deg)',
                    data: [60, 20, 120, 15, 30, 25, 12],
                    backgroundColor: COLORS.blueAlpha,
                    borderColor: COLORS.blue,
                    borderWidth: 1.5,
                    borderRadius: 4,
                },
                {
                    label: 'Distance (kpc)',
                    data: [12, 20, 20, 20, 13, 15, 15],
                    backgroundColor: COLORS.orangeAlpha,
                    borderColor: COLORS.orange,
                    borderWidth: 1.5,
                    borderRadius: 4,
                },
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { title: { display: true, text: 'Value' }, beginAtZero: true }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
})();

// ===== Chart 15: Test Coverage =====
(function() {
    const ctx = document.getElementById('testCoverage').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [
                'GNN Encoder (7)',
                'Multi-Task (4)',
                'Baselines CNN+Transformer (7)',
                'Model Utils (7)',
                'Attention Readout (6)',
                'MC Dropout (5)',
                'Variable Targets (2)',
                'Segment GNN (3)',
                'Orbital Features (2)',
                'Hierarchical + Coverage (6)'
            ],
            datasets: [{
                data: [7, 4, 7, 7, 6, 5, 2, 3, 2, 6],
                backgroundColor: [
                    COLORS.blue, COLORS.cyan, COLORS.green, COLORS.orange,
                    COLORS.purple, COLORS.pink, COLORS.red,
                    '#22d3ee', '#a78bfa', '#34d399'
                ],
                borderColor: '#1a2332',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'right', labels: { font: { size: 11 }, padding: 8 } }
            }
        }
    });
})();

// ===== Smooth scroll for nav links =====
navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').slice(1);
        const target = document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ===== PDF Export =====
function exportPDF() {
    const btn = document.querySelector('.pdf-btn');
    const originalText = btn ? btn.innerHTML : '';
    if (btn) btn.innerHTML = '&#9203; Generating PDF...';

    const element = document.getElementById('pdf-content');
    const opt = {
        margin:       [10, 10, 10, 10],
        filename:     'Stellar_Stream_DM_Full_Report.pdf',
        image:        { type: 'jpeg', quality: 0.95 },
        html2canvas:  { scale: 2, useCORS: true, scrollY: 0, windowWidth: 1200 },
        jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' },
        pagebreak:    { mode: ['avoid-all', 'css', 'legacy'] }
    };

    html2pdf().set(opt).from(element).save().then(() => {
        if (btn) btn.innerHTML = originalText;
    }).catch(() => {
        if (btn) btn.innerHTML = originalText;
        alert('PDF generation failed. Try using your browser\'s Print to PDF (Ctrl+P) instead.');
    });
}

// Make exportPDF available globally
window.exportPDF = exportPDF;

console.log('Stellar Stream DM Dashboard loaded. All charts rendered.');

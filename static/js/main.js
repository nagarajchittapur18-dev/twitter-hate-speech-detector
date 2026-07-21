/**
 * static/js/main.js
 * Twitter Hate Speech Detector — Interactive JS
 */

'use strict';

/* ══════════════════════════════════════════════════════════════
   1. Character Counter
══════════════════════════════════════════════════════════════ */
function initCharCounter() {
  const ta      = document.getElementById('tweet-input');
  const counter = document.getElementById('char-counter');
  const maxLen  = 500;

  if (!ta || !counter) return;

  function update() {
    const len = ta.value.length;
    counter.textContent = `${len} / ${maxLen}`;
    counter.className   = 'char-counter';
    if (len > maxLen * .85) counter.classList.add('warn');
    if (len >= maxLen)       counter.classList.add('limit');
  }

  ta.addEventListener('input', update);
  update();
}

/* ══════════════════════════════════════════════════════════════
   2. Loading Overlay
══════════════════════════════════════════════════════════════ */
function initPredictForm() {
  const form    = document.getElementById('predict-form');
  const overlay = document.getElementById('loading-overlay');
  const msgs    = [
    '🔍 Preprocessing tweet…',
    '🧠 Running BiLSTM model…',
    '🔎 Computing attention weights…',
    '📊 Generating visualizations…',
    '✅ Finalizing results…',
  ];

  if (!form || !overlay) return;

  form.addEventListener('submit', function (e) {
    const tweet = document.getElementById('tweet-input')?.value.trim();
    if (!tweet) { e.preventDefault(); return; }

    overlay.classList.add('show');
    let i = 0;
    const msgEl = overlay.querySelector('.loader-text');
    if (msgEl) {
      msgEl.textContent = msgs[0];
      const interval = setInterval(() => {
        i = (i + 1) % msgs.length;
        msgEl.textContent = msgs[i];
      }, 900);
      // Safety: remove after 30s
      setTimeout(() => { clearInterval(interval); overlay.classList.remove('show'); }, 30000);
    }
  });
}

/* ══════════════════════════════════════════════════════════════
   3. Confidence Ring Animation
══════════════════════════════════════════════════════════════ */
function initConfidenceRing() {
  const ring = document.querySelector('.ring-fill');
  if (!ring) return;

  const confidence = parseFloat(ring.dataset.confidence || 0);
  const circumference = 340;
  const offset = circumference - (confidence / 100) * circumference;

  setTimeout(() => {
    ring.style.strokeDashoffset = offset;
  }, 300);
}

/* ══════════════════════════════════════════════════════════════
   4. Dashboard Charts (Chart.js)
══════════════════════════════════════════════════════════════ */
function initDashboardCharts() {
  // Pie Chart
  const pieCtx = document.getElementById('pieChart');
  if (pieCtx) {
    const labels = JSON.parse(pieCtx.dataset.labels || '[]');
    const values = JSON.parse(pieCtx.dataset.values || '[]');

    new Chart(pieCtx, {
      type: 'doughnut',
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: ['#ff4757', '#ffa502', '#00ff88'],
          borderColor:      ['#ff4757', '#ffa502', '#00ff88'],
          borderWidth: 2,
          hoverOffset: 8,
        }]
      },
      options: {
        responsive: true,
        cutout: '65%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#8892a4', padding: 16, font: { size: 13 } }
          },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.label}: ${ctx.raw} (${((ctx.raw / values.reduce((a,b)=>a+b,0))*100).toFixed(1)}%)`
            }
          }
        }
      }
    });
  }

  // Bar Chart
  const barCtx = document.getElementById('barChart');
  if (barCtx) {
    const labels = JSON.parse(barCtx.dataset.labels || '[]');
    const values = JSON.parse(barCtx.dataset.values || '[]');

    new Chart(barCtx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Predictions',
          data: values,
          backgroundColor: [
            'rgba(255,71,87,.6)',
            'rgba(255,165,2,.6)',
            'rgba(0,255,136,.6)'
          ],
          borderColor:     ['#ff4757','#ffa502','#00ff88'],
          borderWidth: 2,
          borderRadius: 8,
          borderSkipped: false,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: {
            grid:  { color: 'rgba(255,255,255,.04)' },
            ticks: { color: '#8892a4' }
          },
          y: {
            grid:  { color: 'rgba(255,255,255,.04)' },
            ticks: { color: '#8892a4', precision: 0 },
            beginAtZero: true,
          }
        }
      }
    });
  }

  // Line Chart (confidence trend)
  const lineCtx = document.getElementById('lineChart');
  if (lineCtx) {
    const labels = JSON.parse(lineCtx.dataset.labels || '[]');
    const values = JSON.parse(lineCtx.dataset.values || '[]');

    new Chart(lineCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Confidence %',
          data: values,
          fill: true,
          backgroundColor: 'rgba(0,210,255,.08)',
          borderColor: '#00d2ff',
          borderWidth: 2,
          pointBackgroundColor: '#00d2ff',
          pointRadius: 4,
          tension: 0.4,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#8892a4' } } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,.04)' }, ticks: { color: '#8892a4' } },
          y: {
            grid: { color: 'rgba(255,255,255,.04)' },
            ticks: { color: '#8892a4' },
            min: 0, max: 100,
          }
        }
      }
    });
  }
}

/* ══════════════════════════════════════════════════════════════
   5. Progress Bars Animate on Scroll
══════════════════════════════════════════════════════════════ */
function initProgressBars() {
  const bars = document.querySelectorAll('.progress-bar-animated-custom');
  if (!bars.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const bar = entry.target;
        bar.style.width = bar.dataset.width + '%';
        observer.unobserve(bar);
      }
    });
  }, { threshold: 0.3 });

  bars.forEach(bar => {
    bar.style.width = '0%';
    observer.observe(bar);
  });
}

/* ══════════════════════════════════════════════════════════════
   6. Auto-dismiss Flash Messages
══════════════════════════════════════════════════════════════ */
function initFlashMessages() {
  const alerts = document.querySelectorAll('.flash-alert');
  alerts.forEach(alert => {
    setTimeout(() => {
      alert.style.opacity    = '0';
      alert.style.transform  = 'translateX(100%)';
      alert.style.transition = 'all .4s ease';
      setTimeout(() => alert.remove(), 400);
    }, 4000);
  });
}

/* ══════════════════════════════════════════════════════════════
   7. Probabilities Mini-bars (Result Page)
══════════════════════════════════════════════════════════════ */
function initProbBars() {
  const probBars = document.querySelectorAll('.prob-bar');
  probBars.forEach(bar => {
    const target = parseFloat(bar.dataset.value || 0);
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = target + '%'; }, 400);
  });
}

/* ══════════════════════════════════════════════════════════════
   8. Scroll-reveal (Fade-up elements)
══════════════════════════════════════════════════════════════ */
function initScrollReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('fade-up');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.15 });

  els.forEach(el => observer.observe(el));
}

/* ══════════════════════════════════════════════════════════════
   9. Copy API Snippet
══════════════════════════════════════════════════════════════ */
function copyAPISnippet() {
  const code = document.getElementById('api-snippet');
  if (!code) return;
  navigator.clipboard.writeText(code.textContent).then(() => {
    const btn = document.getElementById('copy-btn');
    if (btn) { btn.textContent = '✅ Copied!'; setTimeout(() => btn.textContent = '📋 Copy', 2000); }
  });
}

/* ══════════════════════════════════════════════════════════════
   10. Init All
══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initCharCounter();
  initPredictForm();
  initConfidenceRing();
  initDashboardCharts();
  initProgressBars();
  initFlashMessages();
  initProbBars();
  initScrollReveal();
});

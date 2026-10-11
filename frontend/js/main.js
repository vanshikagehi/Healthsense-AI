// ═══════════════════════════════════════════
//   HealthSense AI — Shared JavaScript
//   main.js — used across all pages
// ═══════════════════════════════════════════

// ── Navbar scroll effect ──
const navbar = document.querySelector('.navbar');
if (navbar) {
  window.addEventListener('scroll', () => {
    navbar.classList.toggle('scrolled', window.scrollY > 40);
  });
}

// ── Scroll-triggered fade-up for cards ──
const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
      entry.target.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
      fadeObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.card, .unique-card, .workflow-step, .metric-card').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(18px)';
  fadeObserver.observe(el);
});

// ── Animate progress bars when visible ──
const progressObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const fill = entry.target;
      const target = fill.dataset.width || '0';
      fill.style.width = target + '%';
      progressObserver.unobserve(fill);
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.progress-fill').forEach(el => {
  el.style.width = '0%';
  progressObserver.observe(el);
});

// ── Active nav link highlighter ──
function setActiveNavLink() {
  const path = window.location.pathname;
  document.querySelectorAll('.nav-links a, .sidebar-nav a').forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === path || link.getAttribute('href') === path.split('/').pop());
  });
}
setActiveNavLink();

// ── Tab switcher ──
function initTabs(containerSelector) {
  const containers = document.querySelectorAll(containerSelector);
  containers.forEach(container => {
    const tabs = container.querySelectorAll('[data-tab]');
    const panels = container.querySelectorAll('[data-panel]');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        panels.forEach(p => p.style.display = 'none');
        tab.classList.add('active');
        const target = container.querySelector(`[data-panel="${tab.dataset.tab}"]`);
        if (target) target.style.display = 'block';
      });
    });
    if (tabs[0]) tabs[0].click();
  });
}
initTabs('.tab-container');

// ── HITL Approval Modal ──
function showApprovalModal({ title, description, onYes, onNo }) {
  const existing = document.getElementById('hitl-modal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = 'hitl-modal';
  modal.style.cssText = `
    position:fixed; inset:0; z-index:999;
    background:rgba(5,9,20,0.85); backdrop-filter:blur(8px);
    display:flex; align-items:center; justify-content:center; padding:24px;
  `;
  modal.innerHTML = `
    <div style="background:#0d1526; border:1px solid rgba(245,158,11,0.4); border-radius:18px; padding:32px; max-width:440px; width:100%;">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
        <span style="font-size:24px;">🧑‍💼</span>
        <div>
          <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase; color:#f59e0b; margin-bottom:2px;">Human Approval Required</div>
          <div style="font-size:17px; font-weight:800; color:#e8edf5;">${title}</div>
        </div>
      </div>
      <div style="font-size:14px; color:#6b7fa3; margin-bottom:24px; line-height:1.7; padding:14px; background:rgba(245,158,11,0.06); border-radius:10px; border-left:3px solid #f59e0b;">
        ${description}
      </div>
      <div style="display:flex; gap:12px;">
        <button id="hitl-yes" style="flex:1; padding:12px; background:linear-gradient(135deg,#4f8ef7,#7c5cd8); color:#fff; border:none; border-radius:9px; font-size:14px; font-weight:700; cursor:pointer;">
          ✓ Yes, Proceed
        </button>
        <button id="hitl-no" style="flex:1; padding:12px; background:transparent; color:#e8edf5; border:1px solid #1e2d4a; border-radius:9px; font-size:14px; cursor:pointer;">
          ✗ No, Cancel
        </button>
      </div>
      <div style="text-align:center; font-size:11px; color:#6b7fa3; margin-top:14px;">AI will not act without your explicit approval</div>
    </div>
  `;
  document.body.appendChild(modal);
  document.getElementById('hitl-yes').onclick = function() {
    // Visual confirmed state — green, checkmark, disabled
    this.style.background = 'linear-gradient(135deg,#00c96b,#00a855)';
    this.style.transform  = 'scale(0.97)';
    this.textContent      = '✓ Confirmed!';
    this.disabled         = true;
    document.getElementById('hitl-no').disabled = true;
    document.getElementById('hitl-no').style.opacity = '0.4';
    setTimeout(() => { modal.remove(); if (onYes) onYes(); }, 700);
  };
  document.getElementById('hitl-no').onclick = function() {
    this.style.background = 'rgba(239,68,68,0.15)';
    this.style.borderColor = '#ef4444';
    this.style.color       = '#ef4444';
    this.textContent       = '✗ Cancelled';
    this.disabled          = true;
    document.getElementById('hitl-yes').disabled = true;
    document.getElementById('hitl-yes').style.opacity = '0.4';
    setTimeout(() => { modal.remove(); if (onNo) onNo(); }, 500);
  };
}

// ── Toast notification ──
function showToast(message, type = 'success') {
  const colors = {
    success: '#00d4b4', error: '#ef4444',
    warning: '#f59e0b', info: '#4f8ef7'
  };
  const toast = document.createElement('div');
  toast.style.cssText = `
    position:fixed; bottom:28px; right:28px; z-index:9999;
    background:#0d1526; border:1px solid ${colors[type]};
    color:#e8edf5; padding:14px 20px; border-radius:12px;
    font-size:13px; font-weight:600; max-width:320px;
    box-shadow:0 8px 32px rgba(0,0,0,0.4);
    animation: fadeUp 0.3s ease;
    display:flex; align-items:center; gap:10px;
    border-left: 4px solid ${colors[type]};
  `;
  toast.innerHTML = `<span style="color:${colors[type]}; font-size:16px;">${type === 'success' ? '✓' : type === 'error' ? '✗' : type === 'warning' ? '⚠' : 'ℹ'}</span> ${message}`;
  document.body.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.4s'; setTimeout(() => toast.remove(), 400); }, 3000);
}

// ── Confidence bar renderer ──
function renderConfidenceBar(container, percent, label) {
  const color = percent >= 80 ? '#ef4444' : percent >= 60 ? '#f59e0b' : '#00d4b4';
  container.innerHTML = `
    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
      <span style="font-size:12px; color:#6b7fa3;">${label}</span>
      <span style="font-size:14px; font-weight:800; color:${color};">${percent}%</span>
    </div>
    <div style="height:8px; background:#111d35; border-radius:4px; overflow:hidden;">
      <div style="height:100%; width:0%; background:${color}; border-radius:4px; transition:width 1.2s ease;" data-width="${percent}"></div>
    </div>
  `;
  setTimeout(() => {
    const fill = container.querySelector('[data-width]');
    if (fill) fill.style.width = percent + '%';
  }, 100);
}

// ── Export ──
window.HS = { showApprovalModal, showToast, renderConfidenceBar };

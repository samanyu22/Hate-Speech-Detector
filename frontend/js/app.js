// Main Application Controller for ThoughtStream Research Platform
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  loadOverviewStats();
  initDetectionWorkspace();
  loadDatasets();
  loadModels();
  loadOptimizationData();
  loadFairnessAudit();
  loadExperiments();
  loadPaperDraft();
  loadSettings();
});

// 1. Dark Mode Management (Cojeev Theme Toggle)
const SUN_PATH = "M12 5C15.866 5 19 8.134 19 12C19 15.866 15.866 19 12 19C8.134 19 5 15.866 5 12C5 8.134 8.134 5 12 5Z";
const MOON_PATH = "M18.9 15.2C17.2 19.3 12.9 21.2 8.8 19.4C4.7 17.6 2.7 13 4.4 8.8C5.4 6.2 7.7 4.3 10.4 3.8C8.1 10.4 12.3 16.8 18.9 15.2Z";

function updateThemeUI(isDark) {
  const btn = document.getElementById('themeToggleBtn');
  if (!btn) return;
  btn.setAttribute('aria-checked', isDark ? 'true' : 'false');
  btn.setAttribute('title', `Switch to ${isDark ? 'light' : 'dark'} theme`);
  
  const label = btn.querySelector('.v-theme-toggle__label');
  if (label) {
    label.textContent = isDark ? 'Dark' : 'Light';
  }
  
  const bodyPath = btn.querySelector('.v-theme-body');
  if (bodyPath) {
    bodyPath.setAttribute('d', isDark ? MOON_PATH : SUN_PATH);
  }
}

function applyTheme(targetMode, event = null) {
  const isDark = targetMode === 'dark';

  const updateDOM = () => {
    if (isDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
      document.body.classList.add('dark-mode');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
      document.body.classList.remove('dark-mode');
    }
    localStorage.setItem('thoughtstream_theme', targetMode);
    updateThemeUI(isDark);
  };

  const prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (!document.startViewTransition || prefersReduced) {
    updateDOM();
    return;
  }

  const x = event?.clientX ?? window.innerWidth / 2;
  const y = event?.clientY ?? window.innerHeight / 2;
  const endRadius = Math.hypot(
    Math.max(x, window.innerWidth - x),
    Math.max(y, window.innerHeight - y)
  );

  const transition = document.startViewTransition(() => {
    updateDOM();
  });

  transition.ready.then(() => {
    document.documentElement.animate(
      {
        clipPath: [
          `circle(0px at ${x}px ${y}px)`,
          `circle(${endRadius}px at ${x}px ${y}px)`
        ]
      },
      {
        duration: 400,
        easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)',
        pseudoElement: '::view-transition-new(root)'
      }
    );
  }).catch(() => {
    // Graceful fallback if view transition fails
    updateDOM();
  });
}

function makeElementDraggable(el) {
  if (!el) return;

  let isPointerDown = false;
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let initialLeft = 0;
  let initialTop = 0;
  const dragThreshold = 5;
  let suppressClickUntil = 0;

  // Restore saved position if present
  const restorePosition = () => {
    try {
      const savedPos = JSON.parse(localStorage.getItem('thoughtstream_toggle_pos'));
      if (savedPos && typeof savedPos.left === 'number' && typeof savedPos.top === 'number') {
        const btnRect = el.getBoundingClientRect();
        const maxLeft = Math.max(8, window.innerWidth - (btnRect.width || 120) - 8);
        const maxTop = Math.max(8, window.innerHeight - (btnRect.height || 44) - 8);
        const left = Math.min(Math.max(8, savedPos.left), maxLeft);
        const top = Math.min(Math.max(8, savedPos.top), maxTop);
        el.style.position = 'fixed';
        el.style.left = `${left}px`;
        el.style.top = `${top}px`;
        el.style.right = 'auto';
        el.style.bottom = 'auto';
        el.style.zIndex = '999999';
      }
    } catch (_) {}
  };

  restorePosition();

  // If window resizes, keep button in bounds
  window.addEventListener('resize', () => {
    if (el.style.position === 'fixed') {
      const rect = el.getBoundingClientRect();
      const maxLeft = Math.max(8, window.innerWidth - rect.width - 8);
      const maxTop = Math.max(8, window.innerHeight - rect.height - 8);
      const curLeft = parseFloat(el.style.left) || rect.left;
      const curTop = parseFloat(el.style.top) || rect.top;
      el.style.left = `${Math.min(Math.max(8, curLeft), maxLeft)}px`;
      el.style.top = `${Math.min(Math.max(8, curTop), maxTop)}px`;
    }
  });

  el.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return; // primary button only

    isPointerDown = true;
    isDragging = false;
    startX = e.clientX;
    startY = e.clientY;

    const rect = el.getBoundingClientRect();
    initialLeft = rect.left;
    initialTop = rect.top;

    try {
      el.setPointerCapture(e.pointerId);
    } catch (_) {}
  });

  el.addEventListener('pointermove', (e) => {
    if (!isPointerDown) return;

    const dx = e.clientX - startX;
    const dy = e.clientY - startY;

    if (!isDragging) {
      if (Math.hypot(dx, dy) > dragThreshold) {
        isDragging = true;
        el.classList.add('is-dragging');
        el.style.position = 'fixed';
        el.style.zIndex = '999999';
        el.style.right = 'auto';
        el.style.bottom = 'auto';
      }
    }

    if (isDragging) {
      const rect = el.getBoundingClientRect();
      const btnWidth = rect.width || 120;
      const btnHeight = rect.height || 44;
      const maxLeft = Math.max(8, window.innerWidth - btnWidth - 8);
      const maxTop = Math.max(8, window.innerHeight - btnHeight - 8);

      const targetLeft = Math.min(Math.max(8, initialLeft + dx), maxLeft);
      const targetTop = Math.min(Math.max(8, initialTop + dy), maxTop);

      el.style.left = `${targetLeft}px`;
      el.style.top = `${targetTop}px`;
    }
  });

  const onPointerUp = (e) => {
    if (!isPointerDown) return;
    isPointerDown = false;

    if (isDragging) {
      isDragging = false;
      el.classList.remove('is-dragging');
      suppressClickUntil = Date.now() + 250;

      try {
        const rect = el.getBoundingClientRect();
        localStorage.setItem('thoughtstream_toggle_pos', JSON.stringify({
          left: rect.left,
          top: rect.top
        }));
      } catch (_) {}
    }

    try {
      el.releasePointerCapture(e.pointerId);
    } catch (_) {}
  };

  el.addEventListener('pointerup', onPointerUp);
  el.addEventListener('pointercancel', onPointerUp);

  // Click handler: if user was dragging, suppress theme toggle; otherwise toggle theme!
  el.addEventListener('click', (event) => {
    if (Date.now() < suppressClickUntil) {
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    const currentTheme = document.documentElement.getAttribute('data-theme') || (document.body.classList.contains('dark-mode') ? 'dark' : 'light');
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme, event);
  }, true);

  // Double click resets position back to sidebar footer
  el.addEventListener('dblclick', () => {
    el.style.position = '';
    el.style.left = '';
    el.style.top = '';
    el.style.zIndex = '';
    const dock = document.getElementById('themeDock');
    if (dock && !dock.contains(el)) {
      dock.appendChild(el);
    }
    localStorage.removeItem('thoughtstream_toggle_pos');
  });
}

function initTheme() {
  const savedTheme = localStorage.getItem('thoughtstream_theme') || 'light';
  const isDark = savedTheme === 'dark';
  if (isDark) {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.body.classList.add('dark-mode');
  } else {
    document.documentElement.setAttribute('data-theme', 'light');
    document.body.classList.remove('dark-mode');
  }
  updateThemeUI(isDark);

  const toggleBtn = document.getElementById('themeToggleBtn');
  if (toggleBtn) {
    makeElementDraggable(toggleBtn);
  }
}

// 2. Navigation
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  const pages = document.querySelectorAll('.page-view');

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetPage = item.getAttribute('data-target');
      navItems.forEach(n => n.classList.remove('active'));
      pages.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const activePage = document.getElementById(targetPage);
      if (activePage) activePage.classList.add('active');

      // Refresh specific page data on view
      if (targetPage === 'page-datasets') loadDatasets();
      if (targetPage === 'page-models') loadModels();
      if (targetPage === 'page-optimization') loadOptimizationData();
      if (targetPage === 'page-fairness') loadFairnessAudit();
      if (targetPage === 'page-experiments') loadExperiments();
    });
  });
}

// 3. Overview Dashboard
async function loadOverviewStats() {
  try {
    const summary = await API.getSummary();
    document.getElementById('statTotalSamples').innerText = summary.total_samples || '325';
    document.getElementById('statActiveModel').innerText = summary.active_model_name || 'Multilingual Transformer';
    document.getElementById('statActiveVersion').innerText = `v${summary.active_model_version || '1.0'}`;
    document.getElementById('statOptimizationRuns').innerText = summary.optimization_cycles || '0';
    document.getElementById('statTotalFeedback').innerText = summary.total_feedbacks || '0';
    document.getElementById('statCompletedExp').innerText = `${summary.completed_experiments || 0}/9`;
  } catch (e) {
    console.error('Error loading overview:', e);
  }
}

// 4. Detection Workspace
let lastPredictionId = null;

function initDetectionWorkspace() {
  const detectBtn = document.getElementById('btnRunDetection');
  const textInput = document.getElementById('detectTextInput');
  const contextInput = document.getElementById('detectContextInput');
  const resultCard = document.getElementById('detectionResultCard');

  // Preset buttons
  const presets = {
    hate_en: { text: "These vermin immigrants are destroying our country and must be thrown out by force.", ctx: "" },
    sarcasm_counter: { text: "Oh sure, blame every single problem in the world on immigrants, brilliant logic!", ctx: "Replying to xenophobic commentary" },
    hinglish_abuse: { text: "Ye saale kutto jaise bhauk rahe hain, inko raste me goli maar do sabhi ko.", ctx: "" },
    hinglish_civil: { text: "Bhai kal ka cricket match dekh ke maza aa gaya, kya mast sixes maare!", ctx: "Discussion on sports" },
    devanagari_hate: { text: "इन देशद्रोहियों को सरेआम गोली मार देनी चाहिए, ये देश के गद्दार हैं।", ctx: "" },
    devanagari_civil: { text: "त्योहारों पर आपसी भाईचारा और प्रेम बनाए रखना हमारी संस्कृति है।", ctx: "Social greeting" },
    quotation_critique: { text: "Someone wrote 'these immigrants are vermin' which is completely unacceptable racist hate speech.", ctx: "Academic discussion" }
  };

  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.getAttribute('data-preset');
      if (presets[key]) {
        textInput.value = presets[key].text;
        contextInput.value = presets[key].ctx;
      }
    });
  });

  if (detectBtn) {
    detectBtn.addEventListener('click', async () => {
      const text = textInput.value.trim();
      const ctxText = contextInput.value.trim();
      const modelId = document.getElementById('detectModelSelect').value;

      if (!text) {
        alert('Please enter social media post text.');
        return;
      }

      detectBtn.disabled = true;
      detectBtn.innerText = 'Analyzing Multimodal Features...';

      try {
        const payload = {
          text: text,
          context: ctxText ? { thread_history: [ctxText] } : null,
          model_id: modelId,
          explain: true
        };

        const res = await API.detect(payload);
        lastPredictionId = res.prediction_id;
        displayDetectionResult(res);
        loadOverviewStats(); // update counters
      } catch (err) {
        alert(`Detection Error: ${err.message}`);
      } finally {
        detectBtn.disabled = false;
        detectBtn.innerText = 'Execute Detection Pipeline';
      }
    });
  }

  // Feedback Submission
  const submitFeedbackBtn = document.getElementById('btnSubmitFeedback');
  if (submitFeedbackBtn) {
    submitFeedbackBtn.addEventListener('click', async () => {
      if (!lastPredictionId) return;

      const isCorrect = document.getElementById('feedbackCorrectSelect').value === 'true';
      const suggLabel = document.getElementById('feedbackCorrectLabel').value;
      const rating = parseInt(document.getElementById('feedbackRating').value);
      const comment = document.getElementById('feedbackComment').value.trim();

      try {
        const fbPayload = {
          prediction_id: lastPredictionId,
          is_correct: isCorrect,
          suggested_label: suggLabel !== '' ? parseInt(suggLabel) : null,
          explanation_rating: rating,
          comment: comment
        };
        const fbRes = await API.submitFeedback(fbPayload);
        document.getElementById('feedbackStatusMsg').innerText = `Feedback recorded! (Status: ${fbRes.status}). Associated with reward calculation.`;
        loadOptimizationData();
      } catch (e) {
        alert(`Error submitting feedback: ${e.message}`);
      }
    });
  }
}

function displayDetectionResult(res) {
  const card = document.getElementById('detectionResultCard');
  card.style.display = 'block';

  // Label badge
  const badge = document.getElementById('resLabelBadge');
  badge.className = `badge ${res.predicted_label === 1 ? 'badge-hate' : 'badge-nonhate'}`;
  badge.innerText = res.label_name;

  document.getElementById('resConfidence').innerText = `${(res.confidence * 100).toFixed(1)}%`;
  document.getElementById('resLanguage').innerText = res.language_detected.toUpperCase();
  document.getElementById('resModalities').innerText = res.modalities_used.join(', ').toUpperCase();
  document.getElementById('resModelVersion').innerText = `${res.model_id} (v${res.model_version})`;

  // Explainability
  if (res.explanation) {
    const exp = res.explanation;
    document.getElementById('resFaithfulness').innerText = `${(exp.faithfulness_score * 100).toFixed(1)}%`;
    document.getElementById('resNaturalExplanation').innerText = exp.natural_language_explanation;

    // Tokens
    const tokenStream = document.getElementById('resTokenStream');
    tokenStream.innerHTML = '';
    exp.token_attributions.forEach(t => {
      const span = document.createElement('span');
      span.className = `attr-token ${t.polarity === 'positive_evidence' ? 'token-positive' : (t.polarity === 'negative_evidence' ? 'token-negative' : '')}`;
      span.title = `Attribution Score: ${t.score} (${t.polarity})`;
      span.innerText = `${t.token} [${t.score > 0 ? '+' : ''}${t.score}]`;
      tokenStream.appendChild(span);
    });

    // OCR & Context
    document.getElementById('resOcrEvidence').innerText = exp.ocr_evidence || 'No visual text detected.';
    document.getElementById('resContextEvidence').innerText = exp.contextual_evidence ? JSON.stringify(exp.contextual_evidence) : 'None';
  }

  // Scroll smoothly to results
  card.scrollIntoView({ behavior: 'smooth' });
}

// 5. Datasets Laboratory
async function loadDatasets() {
  try {
    const datasets = await API.getDatasets();
    const tbody = document.querySelector('#datasetTable tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    datasets.forEach(ds => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${ds.name}</strong></td>
        <td><code>${ds.id}</code></td>
        <td>${ds.sample_count}</td>
        <td>${Array.isArray(ds.languages) ? ds.languages.join(', ') : ds.languages}</td>
        <td>${Array.isArray(ds.modalities) ? ds.modalities.join(', ') : ds.modalities}</td>
        <td><span class="badge ${ds.sample_count > 0 ? 'badge-nonhate' : 'badge-hate'}">${ds.sample_count > 0 ? 'VALID' : 'EMPTY'}</span></td>
        <td><button class="btn btn-secondary btn-sm" onclick="inspectDataset('${ds.id}')">Inspect</button></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error('Error loading datasets:', e);
  }
}

window.inspectDataset = async function(datasetId) {
  try {
    const data = await API.inspectDataset(datasetId);
    const modal = document.getElementById('datasetInspectModal');
    const container = document.getElementById('datasetInspectContent');
    modal.style.display = 'block';

    let html = `<h4>Dataset: ${datasetId} (${data.total_samples} samples)</h4><div class="table-container" style="max-height: 400px; overflow-y:auto; margin-top:12px;"><table><thead><tr><th>ID</th><th>Text</th><th>Label</th><th>Lang</th></tr></thead><tbody>`;
    data.sample_slice.forEach(s => {
      html += `<tr><td><code>${s.id}</code></td><td>${s.text}</td><td><span class="badge ${s.label === 1 ? 'badge-hate' : 'badge-nonhate'}">${s.label}</span></td><td>${s.language || 'auto'}</td></tr>`;
    });
    html += `</tbody></table></div>`;
    container.innerHTML = html;
  } catch (e) {
    alert(`Failed to inspect dataset: ${e.message}`);
  }
};

// 6. Model Laboratory
async function loadModels() {
  try {
    const data = await API.getModels();
    const tbody = document.querySelector('#modelsTable tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    data.models.forEach(m => {
      const tr = document.createElement('tr');
      const metrics = m.metrics || {};
      const f1 = metrics.macro_f1 ? (metrics.macro_f1 * 100).toFixed(1) + '%' : 'N/A';
      const acc = metrics.accuracy ? (metrics.accuracy * 100).toFixed(1) + '%' : 'N/A';

      tr.innerHTML = `
        <td><strong>${m.name}</strong></td>
        <td><code>${m.id}</code></td>
        <td>v${m.version}</td>
        <td>${m.architecture}</td>
        <td>${acc} / ${f1}</td>
        <td><span class="badge ${m.is_active ? 'badge-nonhate' : 'badge-neutral'}">${m.is_active ? 'ACTIVE' : 'READY'}</span></td>
        <td>${!m.is_active ? `<button class="btn btn-secondary btn-sm" onclick="selectModel('${m.id}')">Activate</button>` : '—'}</td>
      `;
      tbody.appendChild(tr);
    });

    // Populate versions table
    const vbody = document.querySelector('#versionsTable tbody');
    if (vbody) {
      vbody.innerHTML = '';
      data.versions.forEach(v => {
        const vtr = document.createElement('tr');
        vtr.innerHTML = `
          <td><code>${v.version_id}</code></td>
          <td>${v.optimization_method}</td>
          <td>${(v.reward_score * 100).toFixed(1)}%</td>
          <td>${(v.macro_f1 * 100).toFixed(1)}%</td>
          <td>${v.created_at}</td>
        `;
        vbody.appendChild(vtr);
      });
    }
  } catch (e) {
    console.error('Error loading models:', e);
  }
}

window.selectModel = async function(modelId) {
  try {
    await API.selectModel(modelId);
    loadModels();
    loadOverviewStats();
  } catch (e) {
    alert(`Failed to activate model: ${e.message}`);
  }
};

// 7. Reward Optimization Laboratory
async function loadOptimizationData() {
  try {
    const data = await API.getOptimizationHistory();
    const tbody = document.querySelector('#optRunsTable tbody');
    if (tbody) {
      tbody.innerHTML = '';
      data.runs.forEach(r => {
        const tr = document.createElement('tr');
        const beforeF1 = r.metrics_before ? (r.metrics_before.macro_f1 * 100).toFixed(1) + '%' : 'N/A';
        const afterF1 = r.metrics_after ? (r.metrics_after.macro_f1 * 100).toFixed(1) + '%' : 'N/A';

        tr.innerHTML = `
          <td><strong>${r.run_name}</strong></td>
          <td>${r.method}</td>
          <td>${r.feedback_samples_used} pairs</td>
          <td>${beforeF1} → ${afterF1}</td>
          <td><span class="badge badge-nonhate">${r.status}</span></td>
          <td>${r.created_at}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    // Recent rewards
    const rewBody = document.querySelector('#recentRewardsTable tbody');
    if (rewBody) {
      rewBody.innerHTML = '';
      data.recent_rewards.forEach(rew => {
        const rtr = document.createElement('tr');
        rtr.innerHTML = `
          <td><code>${rew.id.substring(0, 8)}</code></td>
          <td>${rew.r_classification}</td>
          <td>${rew.r_explanation}</td>
          <td>${rew.r_context}</td>
          <td>${rew.r_feedback}</td>
          <td><span style="color:var(--color-error)">${rew.r_error}</span></td>
          <td><strong>${rew.total_reward}</strong></td>
        `;
        rewBody.appendChild(rtr);
      });
    }

    // Optimization Trigger button
    const optBtn = document.getElementById('btnTriggerOpt');
    if (optBtn) {
      optBtn.onclick = async () => {
        optBtn.disabled = true;
        optBtn.innerText = 'Optimizing Model via DPO...';
        try {
          const method = document.getElementById('optMethodSelect').value;
          const epochs = parseInt(document.getElementById('optEpochs').value) || 2;
          const res = await API.runOptimization({ method, epochs });
          alert(`Reward Optimization Complete! New Version: ${res.new_version}. Promoted: ${res.promoted_to_active}`);
          loadOptimizationData();
          loadModels();
          loadOverviewStats();
        } catch (err) {
          alert(`Optimization Alert: ${err.message}`);
        } finally {
          optBtn.disabled = false;
          optBtn.innerText = 'Execute Optimization Cycle';
        }
      };
    }
  } catch (e) {
    console.error('Error loading optimization data:', e);
  }
}

// 8. Fairness & Bias Audit
async function loadFairnessAudit() {
  try {
    const report = await API.getFairnessAudit();
    document.getElementById('fairnessOddsGap').innerText = report.equalized_odds_gap ? `${(report.equalized_odds_gap * 100).toFixed(1)}%` : '0.0%';
    document.getElementById('fairnessStatusBadge').innerText = report.is_fair_under_threshold ? 'EQUITABLE PARITY MET' : 'DISPARITY WARNING';
    document.getElementById('fairnessStatusBadge').className = `badge ${report.is_fair_under_threshold ? 'badge-nonhate' : 'badge-hate'}`;

    const tbody = document.querySelector('#fairnessGroupTable tbody');
    if (tbody && report.group_metrics) {
      tbody.innerHTML = '';
      Object.keys(report.group_metrics).forEach(group => {
        const m = report.group_metrics[group];
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${group.replace('_', ' ').toUpperCase()}</strong></td>
          <td>${m.sample_count}</td>
          <td>${(m.accuracy * 100).toFixed(1)}%</td>
          <td>${(m.macro_f1 * 100).toFixed(1)}%</td>
          <td>${(m.fpr * 100).toFixed(1)}%</td>
          <td>${(m.fnr * 100).toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);
      });
    }
  } catch (e) {
    console.error('Error loading fairness audit:', e);
  }
}

// 9. Experiment Laboratory
async function loadExperiments() {
  try {
    const experiments = await API.getExperiments();
    const container = document.getElementById('experimentsContainer');
    if (!container) return;
    container.innerHTML = '';

    experiments.forEach(exp => {
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
          <div>
            <span class="badge badge-neutral" style="margin-bottom:4px;">${exp.code.toUpperCase()}</span>
            <h3 style="margin-top:4px;">${exp.title}</h3>
          </div>
          <button class="btn btn-secondary btn-sm" id="btnRun_${exp.code}" onclick="runExperiment('${exp.code}')">Execute Experiment</button>
        </div>
        <p style="font-size:13px; margin-bottom:12px;">${exp.description}</p>
        <div id="resultBox_${exp.code}" style="display:none; background-color:var(--color-surface-raised); padding:12px; border:1px solid var(--color-border-subtle); margin-top:8px;">
        </div>
      `;
      container.appendChild(div);
    });
  } catch (e) {
    console.error('Error loading experiments:', e);
  }
}

window.runExperiment = async function(expCode) {
  const btn = document.getElementById(`btnRun_${expCode}`);
  const resBox = document.getElementById(`resultBox_${expCode}`);
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Running Pipeline...';
  }

  try {
    const res = await API.runExperiment(expCode);
    resBox.style.display = 'block';
    let html = `<div style="font-family:var(--font-serif); font-weight:700; margin-bottom:6px;">${res.title} — Results</div>`;
    html += `<p style="font-size:13px; color:var(--color-text-primary); margin-bottom:8px;"><strong>Summary:</strong> ${res.summary}</p>`;
    html += `<pre style="font-size:12px; max-height:200px; overflow-y:auto;">${JSON.stringify(res.metrics, null, 2)}</pre>`;
    resBox.innerHTML = html;
    loadOverviewStats();
  } catch (e) {
    alert(`Experiment execution failed: ${e.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = 'Re-Run Experiment';
    }
  }
};

// 10. Paper Draft
async function loadPaperDraft() {
  try {
    const paper = await API.getPaperDraft();
    const pre = document.getElementById('paperTexContent');
    if (pre) pre.innerText = paper.tex_content;
  } catch (e) {
    console.error('Error loading paper draft:', e);
  }
}

// 11. Settings
async function loadSettings() {
  try {
    const s = await API.getSettings();
    document.getElementById('settingDevice').value = s.compute.device || 'cpu';
    document.getElementById('settingBatchSize').value = s.compute.batch_size || 16;
    if (s.reward_weights) {
      document.getElementById('weightAlpha').value = s.reward_weights.alpha || 0.35;
      document.getElementById('weightBeta').value = s.reward_weights.beta || 0.25;
      document.getElementById('weightGamma').value = s.reward_weights.gamma || 0.15;
      document.getElementById('weightDelta').value = s.reward_weights.delta || 0.25;
      document.getElementById('weightLambda').value = s.reward_weights.lambda_err || 0.50;
    }

    const saveBtn = document.getElementById('btnSaveSettings');
    if (saveBtn) {
      saveBtn.onclick = async () => {
        const payload = {
          reward_weights: {
            alpha: parseFloat(document.getElementById('weightAlpha').value),
            beta: parseFloat(document.getElementById('weightBeta').value),
            gamma: parseFloat(document.getElementById('weightGamma').value),
            delta: parseFloat(document.getElementById('weightDelta').value),
            lambda_err: parseFloat(document.getElementById('weightLambda').value)
          }
        };
        await API.updateSettings(payload);
        alert('Settings successfully updated!');
      };
    }
  } catch (e) {
    console.error('Error loading settings:', e);
  }
}


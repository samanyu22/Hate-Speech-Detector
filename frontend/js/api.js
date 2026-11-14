// API Client for Hate Speech Detection Academic Platform
const API = {
  baseUrl: '',

  async getSummary() {
    const res = await fetch(`${this.baseUrl}/api/reports/summary`);
    return res.json();
  },

  async detect(payload) {
    const res = await fetch(`${this.baseUrl}/api/detection/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Detection failed');
    }
    return res.json();
  },

  async submitFeedback(payload) {
    const res = await fetch(`${this.baseUrl}/api/feedback/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Feedback submission failed');
    }
    return res.json();
  },

  async getFeedbackList() {
    const res = await fetch(`${this.baseUrl}/api/feedback/`);
    return res.json();
  },

  async getDatasets() {
    const res = await fetch(`${this.baseUrl}/api/datasets/`);
    return res.json();
  },

  async registerDataset(payload) {
    const res = await fetch(`${this.baseUrl}/api/datasets/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Registration failed');
    }
    return res.json();
  },

  async inspectDataset(datasetId, params = {}) {
    const q = new URLSearchParams();
    if (params.limit != null) q.append('limit', params.limit);
    if (params.offset != null) q.append('offset', params.offset);
    if (params.label_filter != null) q.append('label_filter', params.label_filter);
    if (params.search != null) q.append('search', params.search);
    const qs = q.toString() ? `?${q.toString()}` : '';
    const res = await fetch(`${this.baseUrl}/api/datasets/${datasetId}/inspect${qs}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Inspection failed');
    }
    return res.json();
  },

  async getModels() {
    const res = await fetch(`${this.baseUrl}/api/models/`);
    return res.json();
  },

  async selectModel(modelId) {
    const res = await fetch(`${this.baseUrl}/api/models/select/${modelId}`, { method: 'POST' });
    return res.json();
  },

  async runOptimization(payload) {
    const res = await fetch(`${this.baseUrl}/api/optimization/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Optimization failed');
    }
    return res.json();
  },

  async getOptimizationHistory() {
    const res = await fetch(`${this.baseUrl}/api/optimization/history`);
    return res.json();
  },

  async simulateReward(payload) {
    const res = await fetch(`${this.baseUrl}/api/optimization/simulate-reward`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Reward simulation failed');
    }
    return res.json();
  },

  async getFairnessAudit() {
    const res = await fetch(`${this.baseUrl}/api/fairness/audit`);
    return res.json();
  },

  async runFairnessProbe(payload) {
    const res = await fetch(`${this.baseUrl}/api/fairness/counterfactual-probe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Fairness probe failed');
    }
    return res.json();
  },

  async getExperiments() {
    const res = await fetch(`${this.baseUrl}/api/experiments/`);
    return res.json();
  },

  async runExperiment(expCode) {
    const res = await fetch(`${this.baseUrl}/api/experiments/run/${expCode}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Experiment execution failed');
    }
    return res.json();
  },

  async runAllExperiments() {
    const res = await fetch(`${this.baseUrl}/api/experiments/run-all`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Batch experiment execution failed');
    }
    return res.json();
  },

  async resetExperiments() {
    const res = await fetch(`${this.baseUrl}/api/experiments/reset`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Reset failed');
    }
    return res.json();
  },

  async getPaperDraft() {
    const res = await fetch(`${this.baseUrl}/api/reports/paper-draft`);
    return res.json();
  },

  async getSettings() {
    const res = await fetch(`${this.baseUrl}/api/settings/`);
    return res.json();
  },

  async updateSettings(payload) {
    const res = await fetch(`${this.baseUrl}/api/settings/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return res.json();
  }
};


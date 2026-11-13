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

  async inspectDataset(datasetId) {
    const res = await fetch(`${this.baseUrl}/api/datasets/${datasetId}/inspect`);
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

  async getFairnessAudit() {
    const res = await fetch(`${this.baseUrl}/api/fairness/audit`);
    return res.json();
  },

  async getExperiments() {
    const res = await fetch(`${this.baseUrl}/api/experiments/`);
    return res.json();
  },

  async runExperiment(expCode) {
    const res = await fetch(`${this.baseUrl}/api/experiments/run/${expCode}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Experiment execution failed');
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


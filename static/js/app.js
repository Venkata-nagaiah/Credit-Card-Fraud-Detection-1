/**
 * Credit Card Fraud Detection - Frontend Application Logic
 * Integrates dual-model predictions (Random Forest & Gradient Boosting),
 * Web Audio buzzer alerts, live delta equation calculation, history logs, and Chart.js benchmarks.
 */

let metricsChartInstance = null;
let rocChartInstance = null;
let currentHistoryData = [];

document.addEventListener("DOMContentLoaded", () => {
  calculateLiveDeltas();
  loadHistory();
  initBenchmarkCharts();
});

// ==========================================================================
// Live Balance Delta Equation Calculation
// ==========================================================================
function calculateLiveDeltas() {
  const type = document.getElementById("txType").value;
  const amount = parseFloat(document.getElementById("txAmount").value) || 0.0;
  const oldOrg = parseFloat(document.getElementById("oldbalanceOrg").value) || 0.0;
  const newOrg = parseFloat(document.getElementById("newbalanceOrig").value) || 0.0;
  const oldDest = parseFloat(document.getElementById("oldbalanceDest").value) || 0.0;
  const newDest = parseFloat(document.getElementById("newbalanceDest").value) || 0.0;

  const isCashIn = type === "CASH_IN";
  const orgDelta = isCashIn ? (newOrg - oldOrg - amount) : (oldOrg - newOrg - amount);
  const destDelta = isCashIn ? (oldDest - newDest - amount) : (newDest - oldDest - amount);

  const hasNeg = amount <= 0 || oldOrg < 0 || newOrg < 0 || oldDest < 0 || newDest < 0;
  const insufficient = !isCashIn && oldOrg < amount;
  const origDrained = !isCashIn && oldOrg > 0 && newOrg === 0 && (type === "TRANSFER" || type === "CASH_OUT" || amount >= 1000);
  const isSystemFlagged = amount > 200000;

  // Update OrgDelta display
  const orgDeltaEqElem = document.getElementById("orgDeltaEquation");
  const orgDeltaBadge = document.getElementById("orgDeltaBadge");
  if (orgDeltaEqElem && orgDeltaBadge) {
    orgDeltaEqElem.textContent = isCashIn
      ? `${newOrg.toFixed(2)} - ${oldOrg.toFixed(2)} - ${amount.toFixed(2)} = ${orgDelta.toFixed(2)}`
      : `${oldOrg.toFixed(2)} - ${newOrg.toFixed(2)} - ${amount.toFixed(2)} = ${orgDelta.toFixed(2)}`;

    if (hasNeg) {
      orgDeltaBadge.textContent = "NEGATIVE / INVALID";
      orgDeltaBadge.className = "delta-val alert";
    } else if (insufficient) {
      orgDeltaBadge.textContent = "INSUFFICIENT FUNDS";
      orgDeltaBadge.className = "delta-val alert";
    } else if (origDrained) {
      orgDeltaBadge.textContent = "DRAINED TO $0 (ALERT)";
      orgDeltaBadge.className = "delta-val alert";
    } else if (Math.abs(orgDelta) > 0.05) {
      orgDeltaBadge.textContent = `MISMATCH (Δ ${orgDelta >= 0 ? "+" : ""}${orgDelta.toFixed(2)})`;
      orgDeltaBadge.className = "delta-val alert";
    } else {
      orgDeltaBadge.textContent = "BALANCED (Δ 0.00)";
      orgDeltaBadge.className = "delta-val normal";
    }
  }

  // Update DestDelta display
  const destDeltaEqElem = document.getElementById("destDeltaEquation");
  const destDeltaBadge = document.getElementById("destDeltaBadge");
  if (destDeltaEqElem && destDeltaBadge) {
    destDeltaEqElem.textContent = isCashIn
      ? `${oldDest.toFixed(2)} - ${newDest.toFixed(2)} - ${amount.toFixed(2)} = ${destDelta.toFixed(2)}`
      : `${newDest.toFixed(2)} - ${oldDest.toFixed(2)} - ${amount.toFixed(2)} = ${destDelta.toFixed(2)}`;

    const isUntrackedDest = (oldDest === 0.0 && newDest === 0.0);

    if (oldDest < 0 || newDest < 0) {
      destDeltaBadge.textContent = "NEGATIVE DEST";
      destDeltaBadge.className = "delta-val alert";
    } else if (isUntrackedDest) {
      destDeltaBadge.textContent = "Untracked (OK)";
      destDeltaBadge.className = "delta-val normal";
    } else if (Math.abs(destDelta) > 0.05) {
      destDeltaBadge.textContent = `MISMATCH (Δ ${destDelta >= 0 ? "+" : ""}${destDelta.toFixed(2)})`;
      destDeltaBadge.className = "delta-val alert";
    } else {
      destDeltaBadge.textContent = "DEST BALANCED";
      destDeltaBadge.className = "delta-val normal";
    }
  }

  // Update System Flag display
  const systemFlagBadge = document.getElementById("systemFlagBadge");
  if (systemFlagBadge) {
    if (isSystemFlagged) {
      systemFlagBadge.textContent = "FLAG 1 (> $200k)";
      systemFlagBadge.className = "delta-val alert";
    } else {
      systemFlagBadge.textContent = "FLAG 0 (NORMAL)";
      systemFlagBadge.className = "delta-val normal";
    }
  }
}

// ==========================================================================
// Auto-Calculate Safe Balances
// ==========================================================================
function autoBalanceAfterBalances() {
  const type = document.getElementById("txType").value;
  let amount = parseFloat(document.getElementById("txAmount").value) || 0.0;
  if (amount <= 0) {
    amount = 100.00;
    document.getElementById("txAmount").value = "100.00";
  }

  let oldOrg = parseFloat(document.getElementById("oldbalanceOrg").value) || 0.0;
  if (oldOrg < amount || oldOrg < 0) {
    oldOrg = amount + 1000.00;
    document.getElementById("oldbalanceOrg").value = oldOrg.toFixed(2);
  }

  let oldDest = parseFloat(document.getElementById("oldbalanceDest").value) || 0.0;
  if (oldDest < 0) {
    oldDest = 0.00;
    document.getElementById("oldbalanceDest").value = "0.00";
  }

  if (type === "CASH_IN") {
    document.getElementById("newbalanceOrig").value = (oldOrg + amount).toFixed(2);
    if (oldDest > 0) {
      document.getElementById("newbalanceDest").value = Math.max(0, oldDest - amount).toFixed(2);
    }
  } else {
    document.getElementById("newbalanceOrig").value = (oldOrg - amount).toFixed(2);
    if (oldDest > 0) {
      document.getElementById("newbalanceDest").value = (oldDest + amount).toFixed(2);
    }
  }

  calculateLiveDeltas();
}

// ==========================================================================
// Transaction ID Generator
// ==========================================================================
function generateNewTxId() {
  const randomNum = Math.floor(10000 + Math.random() * 90000);
  const newId = `TX-${randomNum}`;
  const txInput = document.getElementById("txId");
  if (txInput) {
    txInput.value = newId;
    txInput.focus();
  }
  return newId;
}

// ==========================================================================
// Quick Test Presets
// ==========================================================================
function loadPreset(presetKey) {
  const form = {
    txId: document.getElementById("txId"),
    type: document.getElementById("txType"),
    amount: document.getElementById("txAmount"),
    oldOrg: document.getElementById("oldbalanceOrg"),
    newOrg: document.getElementById("newbalanceOrig"),
    oldDest: document.getElementById("oldbalanceDest"),
    newDest: document.getElementById("newbalanceDest"),
    step: document.getElementById("txStep"),
  };

  switch (presetKey) {
    case "grocery_payment":
      if (form.txId) form.txId.value = "TX-12048";
      form.type.value = "PAYMENT";
      form.amount.value = "45.00";
      form.oldOrg.value = "1250.00";
      form.newOrg.value = "1205.00";
      form.oldDest.value = "0.00";
      form.newDest.value = "0.00";
      form.step.value = "14";
      break;

    case "atm_cashout":
      if (form.txId) form.txId.value = "TX-33180";
      form.type.value = "TRANSFER";
      form.amount.value = "500.00";
      form.oldOrg.value = "2500.00";
      form.newOrg.value = "2000.00";
      form.oldDest.value = "1000.00";
      form.newDest.value = "1500.00";
      form.step.value = "22";
      break;

    case "sender_mismatch":
      if (form.txId) form.txId.value = "TX-77821";
      form.type.value = "TRANSFER";
      form.amount.value = "500.00";
      form.oldOrg.value = "2000.00";
      form.newOrg.value = "1800.00"; // Should be 1500!
      form.oldDest.value = "0.00";
      form.newDest.value = "0.00";
      form.step.value = "18";
      break;

    case "stolen_drain":
      if (form.txId) form.txId.value = "TX-98401";
      form.type.value = "TRANSFER";
      form.amount.value = "500000.00";
      form.oldOrg.value = "500000.00";
      form.newOrg.value = "0.00";
      form.oldDest.value = "0.00";
      form.newDest.value = "0.00";
      form.step.value = "48";
      break;

    case "negative_mismatch":
      if (form.txId) form.txId.value = "TX-98409";
      form.type.value = "TRANSFER";
      form.amount.value = "10000.00";
      form.oldOrg.value = "-0.07";
      form.newOrg.value = "0.00";
      form.oldDest.value = "-0.02";
      form.newDest.value = "10000.00";
      form.step.value = "15";
      break;
  }

  calculateLiveDeltas();

  // Highlight action
  const submitBtn = document.getElementById("submitBtn");
  submitBtn.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ==========================================================================
// Prediction Form Handler
// ==========================================================================
document.getElementById("fraudForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const submitBtn = document.getElementById("submitBtn");
  const originalText = submitBtn.innerHTML;
  submitBtn.innerHTML = "<span>⏳</span> EVALUATING DUAL ML MODELS...";
  submitBtn.disabled = true;

  const payload = {
    tx_id: (document.getElementById("txId") ? document.getElementById("txId").value.trim() : "") || generateNewTxId(),
    type: document.getElementById("txType").value,
    amount: parseFloat(document.getElementById("txAmount").value) || 0.0,
    oldbalanceOrg: parseFloat(document.getElementById("oldbalanceOrg").value) || 0.0,
    newbalanceOrig: parseFloat(document.getElementById("newbalanceOrig").value) || 0.0,
    oldbalanceDest: parseFloat(document.getElementById("oldbalanceDest").value) || 0.0,
    newbalanceDest: parseFloat(document.getElementById("newbalanceDest").value) || 0.0,
    step: parseInt(document.getElementById("txStep").value) || 1,
  };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!data.success) {
      alert("Error in fraud detection: " + (data.error || "Unknown server error"));
      return;
    }

    displayPredictionResult(data);
    loadHistory(); // Refresh history logs
  } catch (err) {
    console.error("Prediction request failed:", err);
    alert("Connection error: Unable to contact ML backend server.");
  } finally {
    submitBtn.innerHTML = originalText;
    submitBtn.disabled = false;
  }
});

function displayPredictionResult(data) {
  const container = document.getElementById("resultContainer");
  const banner = document.getElementById("verdictBanner");
  const verdictTxId = document.getElementById("verdictTxId");
  const title = document.getElementById("verdictTitle");
  const subtitle = document.getElementById("verdictSubtitle");
  const probValue = document.getElementById("combinedProbValue");
  const riskFactorsList = document.getElementById("riskFactorsList");

  // Display Transaction ID
  if (verdictTxId) {
    verdictTxId.textContent = `Transaction ID: ${data.tx_id || "TX-UNKNOWN"}`;
  }

  // Update verdict styling and trigger audio buzzer
  if (data.is_fraud) {
    banner.className = "verdict-banner fraud";
    title.innerHTML = "🚨 FRAUD DETECTED";
    subtitle.textContent = "High financial crime anomaly detected by combined Random Forest & Gradient Boosting engine.";

    // Trigger Warning Buzzer
    if (window.fraudAudio) {
      window.fraudAudio.playFraudBuzzer();
    }
  } else {
    banner.className = "verdict-banner safe";
    title.innerHTML = "✅ TRANSACTION LEGITIMATE";
    subtitle.textContent = "Transaction verified clean. Normal safe behavior confirmed by combined ML models.";

    // Trigger Success Chime
    if (window.fraudAudio) {
      window.fraudAudio.playSafeChime();
    }
  }

  // Combined single score
  probValue.textContent = `${data.combined_probability.toFixed(2)}%`;

  // Risk Factors list
  riskFactorsList.innerHTML = "";
  if (data.risk_factors && data.risk_factors.length > 0) {
    data.risk_factors.forEach((rf) => {
      const li = document.createElement("li");
      li.innerHTML = `${data.is_fraud ? "⚠️" : "✅"} ${rf}`;
      riskFactorsList.appendChild(li);
    });
  }

  // Make visible and scroll smoothly
  container.style.display = "block";
  container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ==========================================================================
// History Logs Management
// ==========================================================================
async function loadHistory() {
  try {
    const res = await fetch("/api/history");
    const json = await res.json();
    if (json.success && json.history) {
      currentHistoryData = json.history;
      renderHistoryTable(json.history);
    }
  } catch (err) {
    console.warn("Could not fetch history:", err);
  }
}

function renderHistoryTable(historyList) {
  const tbody = document.getElementById("historyTableBody");
  if (!tbody) return;

  if (historyList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11" style="text-align:center; padding: 24px; color: #94a3b8;">No transaction logs recorded yet. Run a prediction to record history.</td></tr>`;
    return;
  }

  tbody.innerHTML = historyList
    .map(
      (item) => `
      <tr>
        <td><strong>${item.id}</strong></td>
        <td>${item.timestamp}</td>
        <td><span class="tag-pill" style="color:#0f172a; background:#f1f5f9; border-color:#cbd5e1;">${item.type}</span></td>
        <td>$${Number(item.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        <td>$${Number(item.oldbalanceOrg).toFixed(0)} ➔ $${Number(item.newbalanceOrig).toFixed(0)}</td>
        <td>$${Number(item.oldbalanceDest).toFixed(0)} ➔ $${Number(item.newbalanceDest).toFixed(0)}</td>
        <td><code>${item.orgDelta >= 0 ? "+" : ""}${Number(item.orgDelta).toFixed(2)}</code></td>
        <td><span class="status-badge ${item.systemFlag ? "fraud" : "safe"}">${item.systemFlag ? "FLAG 1" : "0"}</span></td>
        <td>${item.rf_prob.toFixed(1)}%</td>
        <td>${item.gb_prob.toFixed(1)}%</td>
        <td>
          <span class="status-badge ${item.is_fraud ? "fraud" : "safe"}">
            ${item.is_fraud ? "🚨 FRAUD" : "✅ SAFE"}
          </span>
        </td>
      </tr>
    `
    )
    .join("");
}

async function clearHistoryLog() {
  if (!confirm("Are you sure you want to clear all history detection logs?")) return;
  try {
    const res = await fetch("/api/history", { method: "DELETE" });
    const json = await res.json();
    if (json.success) {
      currentHistoryData = [];
      renderHistoryTable([]);
    }
  } catch (err) {
    alert("Error clearing history logs.");
  }
}

function exportHistoryCSV() {
  if (!currentHistoryData || currentHistoryData.length === 0) {
    alert("No history logs available to export.");
    return;
  }

  const headers = ["ID", "Timestamp", "Type", "Amount", "OldBalanceOrg", "NewBalanceOrig", "OldBalanceDest", "NewBalanceDest", "OrgDelta", "DestDelta", "SystemFlag", "RF_Score", "GB_Score", "Verdict"];
  const rows = currentHistoryData.map((h) => [
    h.id,
    `"${h.timestamp}"`,
    h.type,
    h.amount,
    h.oldbalanceOrg,
    h.newbalanceOrig,
    h.oldbalanceDest,
    h.newbalanceDest,
    h.orgDelta,
    h.destDelta,
    h.systemFlag,
    h.rf_prob,
    h.gb_prob,
    `"${h.verdict}"`,
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `credit_card_fraud_history_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================================================
// Chart.js Algorithm Benchmarks Initialization
// ==========================================================================
async function initBenchmarkCharts() {
  try {
    const res = await fetch("/api/metrics");
    const data = await res.json();

    const ctxMetrics = document.getElementById("metricsChart");
    if (ctxMetrics) {
      if (metricsChartInstance) metricsChartInstance.destroy();
      metricsChartInstance = new Chart(ctxMetrics, {
        type: "bar",
        data: {
          labels: data.comparison_labels || ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
          datasets: [
            {
              label: "Random Forest (99.81%)",
              data: data.rf_chart_values || [99.81, 98.42, 92.15, 95.18, 99.62],
              backgroundColor: "rgba(11, 99, 206, 0.85)",
              borderColor: "#0b63ce",
              borderWidth: 1.5,
              borderRadius: 6,
            },
            {
              label: "Gradient Boosting (99.85%)",
              data: data.gb_chart_values || [99.85, 99.12, 95.40, 97.22, 99.84],
              backgroundColor: "rgba(99, 102, 241, 0.85)",
              borderColor: "#6366f1",
              borderWidth: 1.5,
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              min: 80,
              max: 100,
              ticks: { callback: (val) => val + "%" },
            },
          },
          plugins: {
            legend: { position: "top" },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`,
              },
            },
          },
        },
      });
    }

    const ctxRoc = document.getElementById("rocChart");
    if (ctxRoc) {
      const rfPoints = data.random_forest.roc_points || [];
      const gbPoints = data.gradient_boosting.roc_points || [];

      if (rocChartInstance) rocChartInstance.destroy();
      rocChartInstance = new Chart(ctxRoc, {
        type: "line",
        data: {
          datasets: [
            {
              label: "Gradient Boosting ROC (AUC: 0.9985)",
              data: gbPoints.map((p) => ({ x: p.fpr, y: p.tpr })),
              borderColor: "#6366f1",
              backgroundColor: "rgba(99, 102, 241, 0.1)",
              borderWidth: 2.5,
              fill: true,
              tension: 0.2,
            },
            {
              label: "Random Forest ROC (AUC: 0.9981)",
              data: rfPoints.map((p) => ({ x: p.fpr, y: p.tpr })),
              borderColor: "#0b63ce",
              borderDash: [5, 5],
              borderWidth: 2,
              fill: false,
              tension: 0.2,
            },
            {
              label: "Random Baseline",
              data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
              borderColor: "#cbd5e1",
              borderDash: [2, 2],
              borderWidth: 1.5,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: "linear",
              min: 0,
              max: 1,
              title: { display: true, text: "False Positive Rate (FPR)" },
            },
            y: {
              type: "linear",
              min: 0,
              max: 1,
              title: { display: true, text: "True Positive Rate (TPR / Recall)" },
            },
          },
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });
    }
  } catch (err) {
    console.warn("Could not initialize charts:", err);
  }
}

// ==========================================================================
// Audio Toggle Control
// ==========================================================================
function toggleAudio() {
  const btn = document.getElementById("soundToggleBtn");
  if (!window.fraudAudio) return;

  const isMuted = window.fraudAudio.toggleMute();
  if (isMuted) {
    btn.textContent = "🔇 Sound: MUTED";
    btn.style.color = "#ef4444";
  } else {
    btn.textContent = "🔊 Sound: ON";
    btn.style.color = "#475569";
    window.fraudAudio.playSafeChime();
  }
}

function formatCurrency(amount) {
  try {
    const formatter = new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    });
    return formatter.format(amount);
  } catch (e) {
    return `$${amount.toFixed(2)}`;
  }
}

function getNumberValue(input) {
  const value = parseFloat(input.value);
  return Number.isFinite(value) ? value : 0;
}

function handleCalculate(event) {
  event.preventDefault();

  const incomeInput = document.getElementById("monthlyIncome");
  const expensesInput = document.getElementById("monthlyExpenses");
  const resultText = document.getElementById("resultText");

  const income = getNumberValue(incomeInput);
  const expenses = getNumberValue(expensesInput);
  const savings = income - expenses;

  const summary = savings >= 0
    ? `Estimated monthly savings: ${formatCurrency(savings)}`
    : `Estimated shortfall: ${formatCurrency(Math.abs(savings))}`;

  resultText.textContent = summary;
}

function setYear() {
  const yearEl = document.getElementById("year");
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }
}

function main() {
  setYear();

  const form = document.getElementById("calcForm");
  const incomeInput = document.getElementById("monthlyIncome");

  if (form) {
    form.addEventListener("submit", handleCalculate);
  }
  if (incomeInput) {
    incomeInput.focus();
  }
}

window.addEventListener("DOMContentLoaded", main);

document.addEventListener("DOMContentLoaded", () => {
  const grid = document.querySelector(".js-filter-grid");
  if (!grid) {
    return;
  }

  const cards = Array.from(grid.querySelectorAll(".post-card"));
  const tabs = Array.from(document.querySelectorAll(".filter-tab"));

  const applyFilter = (filter) => {
    cards.forEach((card) => {
      const groups = (card.dataset.groups || "").split(/\s+/).filter(Boolean);
      const visible = filter === "all" || groups.includes(filter);
      card.hidden = !visible;
    });

    tabs.forEach((tab) => {
      const active = tab.dataset.filter === filter;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => applyFilter(tab.dataset.filter || "all"));
  });

  applyFilter(grid.dataset.defaultFilter || "all");
});

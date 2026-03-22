if (typeof window !== "undefined") {
  const renderMermaid = () => {
    if (typeof mermaid === "undefined") {
      return;
    }
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: document.body.dataset.mdColorScheme === "slate" ? "dark" : "default",
    });
    for (const element of document.querySelectorAll(".mermaid")) {
      element.removeAttribute("data-processed");
    }
    mermaid.run({ querySelector: ".mermaid" });
  };

  if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
    document$.subscribe(renderMermaid);
  } else {
    window.addEventListener("load", renderMermaid);
  }
}

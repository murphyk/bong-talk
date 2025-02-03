export default {
  math: {
    macros: {
      "\\R": "\\mathbb{R}",  // \R now represents \mathbb{R}
      "\\vect": ["\\mathbf{#1}", 1] // Example: \vect{x} → \mathbf{x}
    }
  }
};

// Pluggable pre-compensation methods. Each method gets the Engine (FFT, kernels A/B, target spectrum,
// state X/Y) and decides how to (re)initialise and iterate. To add a method: push an entry here;
// the UI selector is built from this list. Engine primitives: E._wienerInit(), E.fistaStep(), E.params.
export const METHODS = [
  { id: "fista", label: "Otimização com limites (FISTA)", maxIters: 60,
    init: (E) => E._wienerInit(),                     // Wiener + clip as warm start
    step: (E) => E.fistaStep() },
  { id: "wiener", label: "Wiener + limite (rápido)", maxIters: 0,
    init: (E) => E._wienerInit(), step: () => {} },
];
export const methodById = (id) => METHODS.find(m => m.id === id) || METHODS[0];

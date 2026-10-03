# Visão Clara — leitura sem óculos (demo PWA)

**Não substitui exame oftalmológico.** Demonstração experimental de *pré-compensação óptica da tela*:
a imagem exibida é pré-distorcida para o grau (esférico, cilíndrico, eixo), a distância e o ângulo de visão,
para que a imagem formada na retina fique mais nítida.

* Grau → Zernike OSA (Z2^0, Z2^±2) → PSF policromática (R/G/B + aberração cromática do olho) por óptica de Fourier, em WebGL2.
* Distância/pose pela câmera frontal (MediaPipe Face Landmarker, tudo no aparelho, nada é enviado); padrão 30 cm.
* Solver: otimização com limites [0,1] (FISTA, 2 FFT/iteração, alvo com contraste reduzido, *warm start*) e Wiener + limite como inicialização/alternativa.
* Binocular / faixa de distância: mínimos quadrados conjuntos sobre vários PSFs (A = Σw|H|², B = Σw H*).
* Limites físicos: ganho típico ≈ 1 linha de acuidade para 0,5–1,5 D de borrão residual; acima de ~2 D não se recuperam letras pequenas.

Abra no celular, “Adicionar à Tela de Início” para instalar.

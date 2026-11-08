"""
explanations.py
===============
Short, student-friendly explanations (Markdown + LaTeX) displayed inside the app.
Kept in a separate file so app.py stays readable.
"""

HISTOGRAM = """
The **histogram** counts how many pixels have each intensity level (0 = black ... 255 = white).

* Histogram bunched on the **left** -> dark image; bunched on the **right** -> bright image.
* **Narrow** histogram -> low contrast (washed-out look); **wide** histogram -> high contrast.
* A histogram says nothing about *where* the pixels are, only *how many* there are of each level.
* For colour images each of the R, G and B channels has its own histogram.
"""

ENHANCEMENT = {
    "Image negative": r"""
$$s = (L-1) - r \qquad (L = 256)$$
Every intensity is reversed: black becomes white and white becomes black. Useful for viewing
light details on dark backgrounds (e.g. medical images). Applying it twice gives back the original.
""",
    "Log transformation": r"""
$$s = c\,\log(1+r), \qquad c = \frac{255}{\log(256)}$$
Expands dark values and compresses bright values, so dark detail becomes visible.
The constant *c* makes input 255 map to output 255. `c scale` multiplies this *c*.
""",
    "Gamma (power-law) transformation": r"""
$$s = (L-1)\left(\frac{r}{L-1}\right)^{\gamma}$$
* $\gamma < 1$ -> brightens the image (expands dark levels).
* $\gamma > 1$ -> darkens the image (expands bright levels).
* $\gamma = 1$ -> no change. Gamma must be greater than 0.
""",
    "Contrast stretching": r"""
$$s = \frac{r - r_{min}}{r_{max} - r_{min}} \times 255$$
Linearly spreads the used intensity range over the full 0..255 range. With *percentile clipping*
the darkest/brightest few percent of pixels are ignored when choosing $r_{min}, r_{max}$, which
stops one stray pixel from limiting the stretch. If the image has a single intensity
($r_{max}=r_{min}$) the formula would divide by zero, so the image is returned unchanged.
""",
    "Histogram equalization": r"""
$$s_k = \operatorname{round}\!\left(\frac{\mathrm{CDF}(k)-\mathrm{CDF}_{min}}{N-\mathrm{CDF}_{min}}\,(L-1)\right)$$
The cumulative histogram (CDF) is used as the mapping function, which spreads the most frequent
intensities apart and flattens the histogram. For colour images only the **luminance (Y)** channel is
equalised, so colours do not shift. Equalisation can over-amplify noise in flat regions.
""",
    "Gray-level slicing": r"""
$$s = \begin{cases} \text{highlight} & A \le r \le B \\ r \text{ (or 0)} & \text{otherwise} \end{cases}$$
Highlights a chosen intensity band $[A, B]$. Either keep the rest of the image unchanged, or set it
to black to get a mask of that band.
""",
    "Bit-plane slicing": r"""
$$\text{plane}_k = \left\lfloor \frac{r}{2^k} \right\rfloor \bmod 2$$
An 8-bit pixel is 8 binary digits. **Plane 7** (most significant) carries the main shape of the
picture; **plane 0** (least significant) looks like random noise. Rebuilding the image from only the
top planes gives a close approximation, which is the idea behind lossy compression.
""",
}

SPATIAL = {
    "Arithmetic mean": r"""
$$\hat f(x,y) = \frac{1}{mn}\sum_{(s,t)\in S_{xy}} g(s,t)$$
Average of the neighbourhood. Smooths Gaussian-like noise but blurs edges and handles salt-and-pepper noise badly.
""",
    "Median": r"""
$$\hat f(x,y) = \operatorname{median}_{(s,t)\in S_{xy}}\{g(s,t)\}$$
Middle value of the sorted neighbourhood. Excellent for **salt-and-pepper** noise because extreme
values never become the middle value; it also preserves edges better than the mean filter.
""",
    "Maximum": r"""
$$\hat f(x,y) = \max_{(s,t)\in S_{xy}}\{g(s,t)\}$$
Finds the brightest point in the neighbourhood. Removes **pepper** (black) noise but makes
salt noise grow and the image brighter.
""",
    "Minimum": r"""
$$\hat f(x,y) = \min_{(s,t)\in S_{xy}}\{g(s,t)\}$$
Finds the darkest point. Removes **salt** (white) noise but makes pepper noise grow and the image darker.
""",
    "Midpoint": r"""
$$\hat f(x,y) = \tfrac12\left[\max\{g\} + \min\{g\}\right]$$
Average of the largest and smallest values. Works best for randomly distributed noise such as Gaussian
or uniform noise; poor for strong impulse noise.
""",
    "Alpha-trimmed mean": r"""
$$\hat f(x,y) = \frac{1}{mn-d}\sum_{(s,t)\in S_{xy}} g_r(s,t)$$
Sort the $mn$ values, delete the $d/2$ lowest and $d/2$ highest, and average the rest.
$d = 0$ gives the arithmetic mean; $d = mn-1$ gives the median. So it is a blend of both ideas and
copes with a mixture of impulse and Gaussian noise. Valid $d$: even, between 0 and $mn-1$.
""",
    "Weighted mean": r"""
Like the mean filter but the centre pixel gets the largest weight (binomial weights, sum = 1). It blurs less
harshly than the plain average and approximates Gaussian smoothing.
""",
}

EDGE = {
    "Sobel X (horizontal gradient)": r"""
$$G_x = \begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix} * f$$
Measures how fast intensity changes **from left to right**, so it highlights **vertical edges**.
Negative values are shown as their absolute value, scaled so the largest becomes 255.
""",
    "Sobel Y (vertical gradient)": r"""
$$G_y = \begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix} * f$$
Measures change **from top to bottom**, so it highlights **horizontal edges**.
""",
    "Sobel gradient magnitude": r"""
$$|\nabla f| = \sqrt{G_x^2 + G_y^2}$$
Combines both directions into one edge-strength map (independent of edge orientation).
""",
    "Laplacian (edges)": r"""
$$\nabla^2 f = f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)$$
A **second-derivative** operator: it responds to rapid intensity change in all directions
and is zero on flat areas and on smooth ramps. It is very sensitive to noise.
""",
    "Laplacian sharpening": r"""
$$g(x,y) = f(x,y) - c\,\nabla^2 f(x,y)$$
Subtracts the Laplacian (the minus sign is because the kernel centre is negative), which adds
contrast at edges and makes the image look sharper.
""",
    "Unsharp masking": r"""
$$g_{mask} = f - f_{blur}, \qquad g = f + k\,g_{mask}\quad (k=1)$$
Blur the image, subtract the blur to get only the fine detail ("mask"), and add that detail back to the original.
""",
    "High-boost filtering": r"""
$$g = f + k\,(f - f_{blur}), \qquad k > 1$$
Same idea as unsharp masking but with $k>1$, which boosts the high-frequency detail more strongly.
Too large a $k$ causes halos and amplifies noise.
""",
}

FREQUENCY_GENERAL = r"""
**Spatial vs frequency domain.** Spatial filtering works directly on pixels by sliding a small mask
over the image (convolution). Frequency filtering first converts the image with the 2-D DFT (computed
with the FFT), **multiplies** the spectrum by a filter $H(u,v)$, and converts back with the inverse FFT:

$$g(x,y) = \mathcal{F}^{-1}\{H(u,v)\,F(u,v)\}$$

By the convolution theorem, multiplication in the frequency domain equals convolution in the spatial
domain, so both views describe the same filtering. The frequency view makes it easy to *see and choose*
which frequencies are kept. Low frequencies = smooth areas; high frequencies = edges and noise.
The log-magnitude spectrum $\log(1+|F|)$ is shown with the zero frequency in the centre.
"""

FREQUENCY = {
    "Ideal": r"""
$$H_{LP}=\begin{cases}1 & D\le D_0\\0 & D>D_0\end{cases}$$
A sharp circular cut. Easy to understand but its sharp edge causes visible **ringing** (ripples) around edges.
""",
    "Butterworth": r"""
$$H_{LP}=\frac{1}{1+(D/D_0)^{2n}}$$
Smooth transition; $H = 0.5$ at $D=D_0$. Higher order $n$ -> sharper cutoff (closer to ideal, more ringing).
Low orders give little or no ringing.
""",
    "Gaussian": r"""
$$H_{LP}=e^{-D^2/(2D_0^2)}$$
Smoothest transition; **no ringing**. $H = 0.607$ at $D = D_0$.
""",
    "highpass_note": r"""
The high-pass version is $H_{HP} = 1 - H_{LP}$. It removes the zero-frequency (average brightness) term, so the
result is mostly near 0 with positive and negative values. For display it is **min-max scaled** (the option below).
""",
}

METRICS = """
**MSE** $=\\frac{1}{MN}\\sum (f - g)^2$ and **PSNR** $=10\\log_{10}\\frac{255^2}{MSE}$ dB compare two images
of the same size. Important limits:

* They need a **reference** image (the "ground truth"). If you only have a noisy photo, there is no reference.
* Lower MSE / higher PSNR means *numerically closer* to the reference, **not necessarily better-looking**.
  A blurred image can score better than a sharper one that a person prefers.
* An enhancement (contrast stretch, equalisation, sharpening) *deliberately changes* the image, so a worse
  PSNR against the original does not mean the enhancement is bad.
* Identical images give MSE = 0 and PSNR = infinity.
"""

"""Original geometric vector glyphs; no external artwork."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,Polygon,Arc
ACTIVE='#A84B20';PASSIVE='#236B8E';LLAMA='#246B83';QWEN='#B56824';GRAY='#6D7378'
KEYS=['pnfp','evertracer','ctcc','iseal','scw','llmprint','reef','huref','awm','zeroprint']
NAMES=['PN-FP','EverTracer','CTCC','iSeal','SCW','LLMPrint','REEF','HuRef','AWM','ZeroPrint']
def panel_title(ax,k):
 ax.set_title(NAMES[KEYS.index(k)],fontweight='bold',fontsize=8.2,pad=8)

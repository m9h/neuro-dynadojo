"""neuro-dynadojo — a DynaDojo/FSLNets-style ground-truth benchmark for M/EEG dynamics,
connectivity recovery, and foundation-model probing."""
__version__ = "0.1.0"

from .generators import (HopfNetworkSystem, KuramotoNetworkSystem, RingWaveSystem,
                         NetsimSystem, kuramoto_simulate, dominant_wavenumber)
from .algorithms import (fc_algorithms, correlation_fc, partialcorr_fc, imag_coherence_fc,
                         wpli_fc, plv_fc, granger_bivariate, dmd_transition)
from .challenges import edge_recovery_auc, directed_edge_auc, wavenumber_consistency
from .bench import run_benchmark, adversarial_search
from .probes import (probe_factor, factor_dataset, representation_probe,
                     bandpower_embed, braindecode_embed)

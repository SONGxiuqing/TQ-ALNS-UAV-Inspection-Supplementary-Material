"""Reproduce manuscript Table 6: M50, M60, M70, M80, L100, L120, L150, L170."""
import src.table6_scalability_supplement as supplement
import src.table6_scalability_main as mainset

def run_subset(module, ids):
    old_debug = module.DEBUG_MODE
    old_ids = set(module.DEBUG_INSTANCE_IDS)
    try:
        module.DEBUG_MODE = True
        module.DEBUG_INSTANCE_IDS = set(ids)
        module.main()
    finally:
        module.DEBUG_MODE = old_debug
        module.DEBUG_INSTANCE_IDS = old_ids

def main():
    run_subset(supplement, {"M50"})
    mainset.main()
    run_subset(supplement, {"L170"})

if __name__ == "__main__":
    main()

import json,glob
for d in sorted(glob.glob("results_e*")):
    M=json.load(open(d+"/metrics.json")); print("==",d,"selected",M.get("selected_model"),M.get("experiment"))
    for k in ["rules_adapted_weak","rf_s2_summary","rf_s2_summary_patch","rf_s1_only","rf_s2_s1"]:
        if k in M:
            m=M[k]; print("  %-22s n=%s grain=%s bal=%.3f auc=%s gP=%.2f gR=%.2f"%(k,m.get("n"),m.get("n_grain"),m["balanced_accuracy"],m.get("roc_auc"),m["precision_grain"],m["recall_grain"]))
    for k in M:
        if "gap" in k or "asof" in k or "as_of" in k: print("  ",k,str(M[k])[:300])
    print("   CI",M.get("ci",{}).get(M.get("selected_model"),{}).get("balanced_accuracy"))
for p in sorted(glob.glob("results_e2*/deep_*json")):
    t=json.load(open(p))["test"]; print(p.split("/")[-1], round(t["balanced_accuracy"],3), round(t.get("roc_auc",0),3))

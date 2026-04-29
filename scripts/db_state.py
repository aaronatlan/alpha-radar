"""État synthétique de la DB de production. Utilisable régulièrement."""
from collections import Counter
from memory.database import (
    init_db, session_scope,
    RawData, Feature, Thesis, Alert, Evaluation, SignalPerformance,
)


def main() -> None:
    init_db()
    with session_scope() as s:
        raws = s.query(RawData.source).all()
        feats = s.query(Feature.feature_name).all()
        print(f"raw_data           : {len(raws)} lignes")
        for src, n in Counter(r[0] for r in raws).most_common():
            print(f"  - {src:20s} : {n}")
        print(f"features           : {len(feats)} lignes")
        for fn, n in Counter(f[0] for f in feats).most_common():
            print(f"  - {fn:20s} : {n}")
        print(f"theses             : {s.query(Thesis).count()}")
        print(f"evaluations        : {s.query(Evaluation).count()}")
        print(f"alerts             : {s.query(Alert).count()}")
        print(f"signal_performance : {s.query(SignalPerformance).count()}")


if __name__ == "__main__":
    main()

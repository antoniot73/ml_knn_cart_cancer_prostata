"""Ejecución completa de la práctica KNN/CART para cáncer de próstata."""

from __future__ import annotations

from pathlib import Path

from src.dataset import build_dataset_summary, configure_logging, load_dataset
from src.modeling import (
    build_cart_model,
    build_feature_importance_table,
    build_knn_model,
    build_metrics_table,
    build_train_test_split,
    evaluate_model,
    prepare_features_target,
)
from src.visualization import (
    plot_cart_tree,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_feature_importance,
    plot_roc_curves,
)


def main() -> None:
    """Ejecuta el pipeline completo de la práctica."""
    configure_logging()
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "Prostate_Cancer.csv"
    output_graphs = project_root / "outputs" / "graficas"
    output_tables = project_root / "outputs" / "tablas"
    output_tables.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    summary = build_dataset_summary(df)
    summary.to_csv(output_tables / "resumen_dataset.csv", index=False)

    plot_class_distribution(df, output_graphs / "distribucion_clases.png")
    plot_correlation_matrix(df, output_graphs / "matriz_correlacion.png")

    x_data, y_data, _ = prepare_features_target(df)
    x_train, x_test, y_train, y_test = build_train_test_split(x_data, y_data)

    knn_model = build_knn_model(n_neighbors=5)
    cart_model = build_cart_model(max_depth=4)

    knn_model.fit(x_train, y_train)
    cart_model.fit(x_train, y_train)

    knn_result = evaluate_model("KNN", knn_model, x_test, y_test)
    cart_result = evaluate_model("CART", cart_model, x_test, y_test)
    results = [knn_result, cart_result]

    metrics_table = build_metrics_table(results)
    metrics_table.to_csv(output_tables / "metricas_comparativas.csv", index=False)

    feature_table = build_feature_importance_table(cart_model)
    feature_table.to_csv(output_tables / "importancia_variables_cart.csv", index=False)

    plot_feature_importance(feature_table, output_graphs / "importancia_variables_cart.png")
    plot_confusion_matrix(knn_result, output_graphs / "matriz_confusion_knn.png")
    plot_confusion_matrix(cart_result, output_graphs / "matriz_confusion_cart.png")
    plot_roc_curves(results, output_graphs / "curva_roc_comparativa.png")
    plot_cart_tree(cart_model, output_graphs / "arbol_cart.png")

    print("Pipeline ejecutado correctamente.")
    print(metrics_table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

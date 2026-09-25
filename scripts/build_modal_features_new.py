"""Build NEW modal features: python -m scripts.build_modal_features_new --help."""

if __package__:
    from .build_modal_features import main
else:
    from build_modal_features import main


if __name__ == "__main__":
    main(default_backend="hybrid")

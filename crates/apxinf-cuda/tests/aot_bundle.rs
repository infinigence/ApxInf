#[path = "../aot/bundle.rs"]
mod bundle;

#[path = "../src/kernels/fixed_profile.rs"]
mod fixed_profile;

#[test]
fn fixed_language_geometry_matches_native_and_export_recipe() {
    let tokens = fixed_profile::FIXED_SCENE_TOKENS;
    assert!(include_str!("../adapters/fixed_profile.h")
        .contains(&format!("APXINF_FIXED_SCENE_TOKENS = {tokens};")));
    let recipes: serde_json::Value =
        serde_json::from_str(include_str!("../aot/manifest.json")).unwrap();
    let language = recipes["kernels"]
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["id"] == "language-attention")
        .unwrap();
    assert_eq!(language["contract"]["q"][1], tokens);
}

#[test]
fn qwen38_recipe_matches_the_reviewed_exporter() {
    let recipes: serde_json::Value =
        serde_json::from_str(include_str!("../aot/qwen38.json")).unwrap();
    let recipe = &recipes["kernels"][0];
    assert_eq!(recipe["id"], "qwen38-dense-swiglu-nvfp4");
    assert_eq!(recipe["contract"]["tokens"], 2048);
    assert_eq!(recipe["contract"]["hidden"], 5120);
    assert_eq!(recipe["contract"]["output"], 17408);
    let exporter = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("aot/exporters")
        .join(recipe["exporter"].as_str().unwrap());
    assert_eq!(
        bundle::digest(&exporter).unwrap(),
        recipe["exporter_sha256"].as_str().unwrap()
    );
}

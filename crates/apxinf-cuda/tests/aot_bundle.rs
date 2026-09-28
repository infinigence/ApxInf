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

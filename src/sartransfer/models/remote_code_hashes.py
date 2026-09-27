"""SHA-256 of every C-RADIOv4-H remote-code file, recorded after a manual review on 2026-09-25.
EncoderSpec pins the revision; encoders.verify_remote_code refuses any other file content."""

REMOTE_CODE_HASHES = {
    "nvidia/C-RADIOv4-H": {
        "adaptor_attn.py": "baa112b58158d75077e9bba8baba51b2c75f6a6deed2d8046aa5e5f2d5cb669a",
        "adaptor_base.py": "f9d285cc76e70f977f9c78e748f610765f0bd068a13f86d804c07c765cece577",
        "adaptor_generic.py": "3d1b5a015ff3754a9eb64cd56b4d04cae01e537a1ffa606adaf904c55fff07a1",
        "adaptor_mlp.py": "e75a9ed1f7f58c749dec5a510d699f36f32fdb7f3b61cd8a7e3ccd5afd2c9b77",
        "adaptor_module_factory.py": "2beac17dc5824c7d9f925dc782cdd4b49bd15cf206bf1cbaa9108a8966e33316",
        "adaptor_registry.py": "16963e07de300e5f3cbd5e343aed111d04e4e55a50e07f50ccc2ab930883ea1e",
        "cls_token.py": "64ee35f128380c3bb9bc1a913e98ac328220be88964f2092686c637ef866fc79",
        "common.py": "6cc79e1f7b61f4917602d2efc9e0a833fa74d60563f6aac74e4625de23723bab",
        "dinov2_arch.py": "5433d08522c28f46a69e710da1a86fc1d76fc03562dbb9b2b49baf5db21195d3",
        "dual_hybrid_vit.py": "86853ae2ed28c902529cd9b78201350e7dba1cf54ea9e57578241ecc0d88d867",
        "enable_cpe_support.py": "3c1e3df656960f5c83b38ba5f632fc5b335edf1d6de67731396a865aaa5ce1ed",
        "enable_damp.py": "7e2a0b6f8934ac6e4a4768e5ca84cb4ff09ba3726a470f1cf046bd5169b2e334",
        "enable_spectral_reparam.py": "b8e703400604ee5f9cdfac0de9b857b7f5126aad8bba03604b04f52f7641d6cb",
        "eradio_model.py": "bf76c60b991813eef45c76246cd7dcee3fefbf39bde3d2b6ac8576067012d05f",
        "extra_models.py": "a938b96715266d2f65325e56d0d8e42465d23e944e905d2820023472785c55a9",
        "extra_timm_models.py": "d7364d5abf5b617782366948f0e593d7a2df275b076094afa1ebc5276a9f7102",
        "feature_normalizer.py": "a74401f5220c2e5960d39addce9ba40f481efca9ab44bb9d96535ed23e1ef62b",
        "forward_intermediates.py": "eb5a562d11edc655876eeb95bc1c3b4c7484b963dae41a3f76f7acd2a6906b9e",
        "hf_model.py": "d28c0a0238ca8854a9af5a9f003761332bfc9182601b3b7466d9242be91ee96f",
        "input_conditioner.py": "72062d3b71877e1469e069cf090e5f35d6ad7abaeb451e56502f7f32dc6c1075",
        "open_clip_adaptor.py": "c2eb9166de48f5eb5f90f8a750e8ebe67da93698e794dc0bbdab5f013bab8e66",
        "radio_model.py": "2be659f5cd5b9db9c49db70387b65253c74bc16b3eb86e976f0f693271d4b183",
        "siglip2_adaptor.py": "1f40f7d77beefd69ad86aac61b12c82ed915b4012e6bea1628fc8abe80782851",
        "utils.py": "3b83163ce2650ef04973538dd79716ab015d24344ea25d52cfb0c37b093ba4ba",
        "vit_patch_generator.py": "092e9a0b8eca4fee162484ef3ef77b399f686e5e0ffc7be85fb2d69bc2f9ebd9",
        "vitdet.py": "5b83b4c7b90b797d854a6da2fd4093ce9210a52533d3f7ba6bf505793b6c68c3"
    },
}

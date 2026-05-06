import pypff
import os

os.chdir('./example-data')

# Read hk.pff using modernized io2
print("--- Reading HK ---")
hk_pff = pypff.io2.hkpff('hk.pff')
hk_info = hk_pff.readhk()
print(f"Quabo 1019 Temp 1 readings: {hk_info.get('QUABO_1019', {}).get('DET_TEMP', [])[:3]}...")

# Read ph256 data file using PFFSequence
print("\n--- Reading PH256 Data ---")
ph256_seq = pypff.io2.PFFSequence(['start_2023-08-02T00:39:53Z.dp_ph256.bpp_2.module_254.seqno_0.pff'])
print("Dynamically mapping metadata offsets for fast extraction...")
ph256_seq.print_metadata_offsets()
print("\nVerifying offsets against naive JSON parsing...")
ph256_seq.verify_metadata_offsets(num_frames=10)

ph256_data = ph256_seq.get_image_array(count=10)
ph256_md = ph256_seq.get_all_metadata()
print(f"PH256 Data Shape: {ph256_data.shape}")
print(f"First 5 PH256 pkt_nums: {ph256_md.get('pkt_num', [])[:5]}")

# Read img16 data file using PFFSequence
print("\n--- Reading IMG16 Data ---")
img16_seq = pypff.io2.PFFSequence(['start_2023-06-08T04:30:29Z.dp_img16.bpp_2.module_1.seqno_0.pff'])
img16_data = img16_seq.get_image_array(count=10)
img16_md = img16_seq.get_all_metadata()
print(f"IMG16 Data Shape: {img16_data.shape}")
# For module mode, keys are nested (e.g. quabo_0)
print(f"First 5 IMG16 quabo_0 pkt_nums: {img16_md.get('quabo_0', {}).get('pkt_num', [])[:5]}")

# Read config files using PanosetiRun discovery
print("\n--- Reading Configs via PanosetiRun ---")
run = pypff.io2.PanosetiRun('.')
run.show()
print(f"Obs Config Name: {run.configs.get('obs_config').name if 'obs_config' in run.configs else 'None'}")

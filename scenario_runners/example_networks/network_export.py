# On laptop or RPi, in hil_env
import simbench as sb
import pandapower as pp

net = sb.get_simbench_net("1-MV-rural--2-sw")
# pp.to_json(net, "networks/simbench_1_mv_rural_2_sw.json")

# For custom_lv_flat.yaml — export any real or synthetic LV net you want to test with:
net = sb.get_simbench_net("1-LV-semiurb4--2-sw")  # has typed sgens + profiles per memory
pp.to_json(net, "networks/my_lv_feeder.json")
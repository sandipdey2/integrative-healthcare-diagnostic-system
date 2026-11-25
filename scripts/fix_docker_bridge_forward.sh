#!/usr/bin/env sh
# Host fix (not needed on a normal Docker host): let traffic between containers on user-defined Docker bridge
# networks (br-*) pass a stale iptables-legacy FORWARD chain.
#
# Symptom: containers on the default bridge (docker0) can talk, but containers on a compose network time out
# (e.g. Prometheus shows the api target DOWN with "context deadline exceeded").
# Cause: the kernel evaluates BOTH the nftables ruleset (where dockerd, using iptables-nft, adds ACCEPT rules
# for every br-* bridge) and an old iptables-legacy ruleset whose FORWARD policy is DROP and which only has
# rules for docker0. A packet must be accepted by both, so br-* traffic is dropped by the legacy table.
# Fix: mirror Docker's own two br-* forward rules into the legacy DOCKER-USER chain. Docker's nftables rules
# (network isolation, published-port DROP rules) still apply unchanged. Not persistent across reboots.
set -eu
IPT="$(command -v iptables-legacy 2>/dev/null || echo /usr/sbin/iptables-legacy)"
[ -x "$IPT" ] || { echo "iptables-legacy not installed: nothing to do"; exit 0; }
if ! sudo "$IPT" -S FORWARD 2>/dev/null | grep -q -- '-P FORWARD DROP'; then
  echo "legacy FORWARD policy is not DROP: nothing to do"; exit 0
fi
sudo "$IPT" -N DOCKER-USER 2>/dev/null || true
sudo "$IPT" -C FORWARD -j DOCKER-USER 2>/dev/null || sudo "$IPT" -I FORWARD 1 -j DOCKER-USER
sudo "$IPT" -C DOCKER-USER -i br-+ -j ACCEPT 2>/dev/null || sudo "$IPT" -I DOCKER-USER 1 -i br-+ -j ACCEPT
sudo "$IPT" -C DOCKER-USER -o br-+ -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || \
  sudo "$IPT" -I DOCKER-USER 2 -o br-+ -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
echo "legacy DOCKER-USER now:"; sudo "$IPT" -S DOCKER-USER

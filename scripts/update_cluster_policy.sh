DBX_JSON="$(cat config/policy.json)"
DBX_JSON="${DBX_JSON/REPLACE_WITH_POLICY_ID/$CLUSTER_POLICY_ID}"
DBX_JSON="${DBX_JSON/REPLACE_WITH_VERSION/$RELEASE_VERSION}"
DBX_JSON="$(echo $DBX_JSON | tr '\n' ' ')"
if [ -n "${RELEASE_VERSION}" ]; then
    databricks cluster-policies edit --json="$DBX_JSON"
fi
echo "Updated glass_onion policy with latest package version."
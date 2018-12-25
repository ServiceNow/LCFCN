
cd checkpoints

# Download Trancos checkpoint
curl -L https://www.dropbox.com/sh/rms4dg5autwtpnf/AADQBOr1ruFsptbqG_uPt_zCa?dl=0 > trancos_ResFCN.zip
mkdir trancos_ResFCN
unzip trancos_ResFCN.zip -d trancos_ResFCN
rm trancos_ResFCN.zip
mv trancos_ResFCN/State_Dicts/best_model.pth best_model_trancos_ResFCN.pth
mv trancos_ResFCN/history.json history_trancos_ResFCN.json
rm -r trancos_ResFCN

# Download Pascal checkpoint
curl -L https://www.dropbox.com/sh/pwmoej499sfqb08/AABY13YraHYF51yw62Zc1w0-a?dl=0 > pascal_ResFCN.zip
mkdir pascal_ResFCN
unzip pascal_ResFCN.zip -d pascal_ResFCN
rm pascal_ResFCN.zip
mv pascal_ResFCN/State_Dicts/best_model.pth best_model_pascal_ResFCN.pth
mv pascal_ResFCN/history.json history_pascal_ResFCN.json
rm -r pascal_ResFCN

# # Download Shanghai checkpoint
# curl -L https://www.dropbox.com/sh/pwmoej499sfqb08/AABY13YraHYF51yw62Zc1w0-a?dl=0 > pascal_ResFCN.zip
# mkdir pascal_ResFCN
# unzip pascal_ResFCN.zip -d pascal_ResFCN

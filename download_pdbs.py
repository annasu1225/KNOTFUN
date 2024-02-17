'''
This script downloads the PDB files of the proteins in the dataset txt files.
It will download 636 PDB files from the Protein Data Bank and 195 from the AlphaFold Database.
'''

import requests
import pandas as pd

knotprot_cat_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/knotprot_mf_processed.txt"
alphaknot_cat_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/dataset/alphaknot_mf_processed.txt"

# Read the knotprot_knot_type_catalog.txt file
knotprot_df = pd.read_csv(knotprot_cat_path, sep='\t')
# Read the alphaknot_knot_type_catalog.txt file
alphaknot_df = pd.read_csv(alphaknot_cat_path, sep='\t')

# Extract PDB IDs from knotprot_df
pdb_ids = knotprot_df['PDB_code'].tolist()
print(pdb_ids)
print(len(pdb_ids))
# Extract Uniprot IDs from alphaknot_df
uniprot_ids = alphaknot_df['Uniprot_ID'].tolist()
print(uniprot_ids)
print(len(uniprot_ids))

failed_ids = []

# Shared folder path to save the PDB files
shared_folder_path = "/gpfs/gibbs/pi/gerstein/as4272/KnotFun/PDBs"

for pdb_id in pdb_ids:
    # Define the URL for fetching the experimental PDB file
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    # Make an HTTP request to the PDB and save the file in the shared folder
    response = requests.get(pdb_url)
    if response.status_code == 200:
        with open(f"{shared_folder_path}/{pdb_id}.pdb", "wb") as pdb_file:
            pdb_file.write(response.content)
        print(f"Downloaded {pdb_id}.pdb")
    else:
        failed_ids.append(pdb_id)
        print(f"Failed to download {pdb_id}.pdb")

for uniprot_id in uniprot_ids:
    # Define the URL for fetching the predicted PDB file
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

    # Make an HTTP request to the PDB and save the file in the shared folder
    response = requests.get(pdb_url)
    if response.status_code == 200:
        with open(f"{shared_folder_path}/{uniprot_id}.pdb", "wb") as pdb_file:
            pdb_file.write(response.content)
        print(f"Downloaded {uniprot_id}.pdb")
    else:
        failed_ids.append(uniprot_id)
        print(f"Failed to download {uniprot_id}.pdb")

print("Download completed.")
print(failed_ids)
print(len(failed_ids))

# List of PDB IDs and UniProt IDs in the dataset txt files
# pdb_ids = ['1uak', '1by7', '1cmx', '1f48', '1fug', '1gz0', '1js1', '1mxa', '1mxb', '1mxc', '1mxi', '1ns5', '1nxz', '1o6d', '1o90', '1o92', '1o93', '1o9t', '1p7l', '1p9p', '1qm4', '1rg9', '1to0', '1uaj', '1ual', '1uam', '1v2x', '1v6z', '1vh0', '1vhk', '1vhy', '1x7o', '1x7p', '1xd3', '1xd3', '1xra', '1xrb', '1xrc', '1yrl', '1yrl', '1zjr', '1znc', '1ztu', '2cx8', '2egv', '2egw', '2etl', '2etl', '2fg6', '2g7m', '2ha8', '2i6d', '2o3a', '2o9b', '2o9c', '2obv', '2ool', '2p02', '2qmm', '2qwv', '2v3j', '2v3k', '2vea', '2wdt', '2wdt', '2we6', '2we6', '2yy8', '2z0y', '2znc', '3a7s', '3a7s', '3ai9', '3aia', '3b1b', '3bbd', '3bbe', '3bbh', '3c2w', '3e5y', '3ed7', '3f7b', '3fw3', '3g6o', '3gyq', '3ibr', '3ic6', '3ief', '3ifw', '3ifw', '3ilk', '3iml', '3irt', '3irt', '3jxf', '3jxg', '3jxh', '3knu', '3kty', '3kvf', '3kvf', '3kw2', '3kw5', '3kw5', '3ky7', '3kzc', '3kzk', '3kzn', '3kzo', '3l02', '3l04', '3l05', '3l06', '3l8u', '3m4j', '3m4n', '3m5c', '3m5d', '3mt5', '3n4j', '3n4k', '3nhq', '3nk6', '3nk7', '3o7b', '3oii', '3oij', '3oin', '3onp', '3q98', '3quv', '3rii', '3rii', '3ris', '3ris', '3rv2', '3s7q', '3s82', '3tde', '3ulk', '3ulk', '3znc', '4dm9', '4dm9', '4e04', '4e8b', '4fak', '4fmw', '4gw9', '4h3z', '4h6v', '4ijg', '4jak', '4jwf', '4jwh', '4jwj', '4kdz', '4kgn', '4l69', '4l7i', '4mcd', '4o6d', '4o6d', '3zq5', '4l4q', '4n2x', '4n2x', '4n2x', '1uch', '1uch', '3wj8', '3wj8', '3wj8', '3s7o', '3nou', '3not', '3nop', '3s7p', '3s7n', '4q0i', '4q0h', '4cqh', '4jal', '4jkj', '4jkj', '2len', '2len', '4jkj', '4jkj', '4mcc', '4mcb', '4l2z', '4k0b', '3axz', '4ig6', '4hpv', '4h3y', '3kzm', '4jwg', '4pzk', '4pzk', '4ndn', '4ndn', '4ndn', '4ndn', '3so4', '4ktv', '4ktv', '4ktv', '4ktv', '4ktt', '4ktt', '4ktt', '4ktt', '4o01', '4o01', '4o0p', '4o0p', '4o01', '4o01', '4cng', '4cnf', '4cne', '4cnd', '4j3c', '3cwo', '3ihr', '3ihr', '4q0j', '3s97', '3kld', '4o8g', '3j7y', '3j7y', '4rg1', '4x3m', '4yqd', '4y5f', '4y3i', '4uf6', '4uf6', '4r6l', '4ws9', '4wlr', '4wlr', '4wlq', '4wlq', '4wlp', '4wlp', '4uf5', '4uf5', '4uem', '4uem', '4uel', '4uel', '4xbo', '5e4r', '5e4r', '4yvk', '4yvj', '4yvi', '4yvh', '4yvg', '5a7y', '5fai', '5c77', '4z1w', '4zrr', '4x3l', '5c74', '5a7z', '5a7t', '5a1i', '5co4', '5apg', '5ap8', '4yqt', '4yqs', '4yqr', '4ypx', '5jpq', '5i5l', '5hsq', '5e5r', '5syb', '5t8s', '5t8t', '5gm8', '5gmc', '5gmb', '5ife', '5k5b', '5l8m', '5lbr', '5h9u', '5h5f', '5h5e', '5ipz', '5twk', '5twj', '5vik', '5gra', '5oom', '5oql', '5o95', '5o96', '5wyr', '5wyq', '5nfx', '5nm3', '5nwn', '5l0z', '5kzk', '5ku6', '6fht', '1j85', '5a19', '5vm8', '5ugh', '1cng', '4rq9', '5dog', '5doh', '5lle', '5llg', '3mhl', '3m5e', '3m40', '3m3x', '3m67', '3m98', '3m2n', '3mhm', '3m96', '5drs', '5l6k', '5llo', '5lln', '5llp', '5msb', '3po6', '5thj', '5jn8', '5jna', '5jnc', '5t75', '5m78', '5y2r', '5y2s', '5dsr', '1hva', '3bet', '3caj', '3ibu', '3ibn', '3ibl', '5dsq', '5dso', '5g03', '5sz0', '5sz1', '5sz2', '5sz3', '5sz4', '5sz7', '5jn3', '5nxg', '5nxi', '5nxo', '5nxp', '5nxv', '5nxw', '5ny1', '5ny3', '5ny6', '5nya', '1fsn', '2cbe', '3rz5', '3rz7', '3s76', '1zsa', '6emu', '5viq', '5viv', '1cnh', '5c5k', '4xtq', '4twm', '4twl', '5mg0', '5wyk', '5a1g', '4cnd', '5nrl', '5o9z', '5gmn', '2ax2', '2fos', '1fqm', '1xev', '2nxt', '5l70', '3czv', '3okv', '3sbi', '3bl0', '3rz0', '3t85', '5yuj', '2osf', '4kap', '1g0e', '2ez7', '1if9', '1if7', '2cbc', '4lu3', '1thk', '3s73', '6got', '2wej', '3k34', '3fr7', '3fr7', '4mlt', '5eij', '4r59', '4mdg', '3mnu', '2nno', '1cao', '3v2j', '3kon', '4m2r', '2nwo', '5msa', '5fl5', '1yve', '1yve', '1cak', '5flp', '5flr', '4xz5', '3ml5', '4qsb', '5ty1', '1cnj', '3mhc', '5fls', '5umc', '5g01', '4ywp', '4mdl', '4fpt', '3s72', '3daz', '2vvb', '4z1e', '3kkx', '4kp5', '4l5v', '3u3a', '3hkn', '5tfx', '5zya', '2fnn', '1ugd', '3p3j', '5jes', '3gz0', '4qk1', '5flo', '5th4', '5cjf', '1z97', '5sz5', '1fsq', '3ryj', '1ze8', '4ygn', '5lmd', '5jgs', '4kp8', '4z1k', '4qrv', '2pou', '6b00', '5u0e', '5g0b', '5ll9', '4m2u', '4r5a', '2fnk', '5fnm', '4iwz', '4ht2', '3s74', '3v7x', '3dv7', '4qjm', '6bbs', '4mdm', '3n0n', '4q8z', '4fl7', '5fdc', '5vgy', '2wd3', '4e5q', '4wl4', '5u0f', '4kuy', '3v2m', '1g0f', '5tya', '4pyx', '6fe1', '3ryx', '1raz', '4e4a', '2fnm', '3r16', '5wlr', '6d1l', '3vbd', '5jeh', '4q90', '2nns', '3myq', '3m1q', '2eu2', '4e3f', '4e3g', '2x7t', '2ili', '4fik', '3p55', '2nwz', '3uyn', '5t71', '3ml2', '4k1q', '5ty8', '5g0c', '1fsr', '6g98', '5dvx', '5amd', '4qef', '4hew', '3oku', '1z93', '5brw', '5wex', '4q06', '3p5a', '5je7', '5ekm', '4z1j', '3oys', '4itp', '3dvb', '2pow', '4knn', '3sax', '4qsa', '3s8x', '6eda', '6eea', '6i0j', '6i0w', '6hr3', '6nj4', '6nj5', '6nj2', '6ql3', '6hzx', '6rqn', '6rqq', '6hc7', '6qqm', '6ptx', '6pu2', '6ptq', '6qrd', '6qrf', '6qrc', '6qqq', '6qra', '6qqr', '6qre', '6qrg', '6qr0', '6qr1', '6qqt', '6qr3', '6qqv', '6qr2', '6qqw', '6qqz', '6ac6', '6rxz', '6rxy', '6rxx', '6rxv', '6rxt', '6afk', '6nd4', '6qx9', '1urt', '6bay', '6bap', '6bao', '6bak', '6baf', '5z57', '5z56', '4cne', '4cnf', '4cng', '6emt', '6emv', '6ems', '6gaw', '2ofz', '6h36', '6qn5', '6qng', '6qnl', '6ugn', '6vj3', '6r6y', '6ux1', '6otm', '6oti', '6oue', '6rg4', '6rhj', '6rof', '6wq7', '6km1', '6km2', '6km3', '6km5', '6y74', '6yma', '2fg7']
# uniprot_ids = ['Q54T48', 'K7K4Y1', 'Q8GWE1', 'Q965V9', 'Q8IKM8', 'Q54N38', 'Q9UUB6', 'Q9WUP7', 'Q9VYQ3', 'Q9Y5K5', 'I1NIJ5', 'I1LBV4', 'A0A1D8PNY8', 'O04482', 'P35127', 'Q6ETK2', 'Q9XZ61', 'I1MV53', 'Q6K6Z5', 'Q4D1R0', 'E9AH53', 'Q9FFF2', 'O23482', 'Q09444', 'D3ZHS6', 'Q7K5N4', 'Q99PU7', 'Q92560', 'A0A0R0K2F0', 'A1L2G3', 'A0A1D6KM35', 'A0A1D6HPV4', 'Q5VPB9', 'Q65XK0', 'A0A1D6FKV6', 'B4FFZ2', 'I1M3M4', 'Q05758', 'I1LSB5', 'I1LUL9', 'P05793', 'K7KXV3', 'I1L257', 'I1MGE5', 'P14713', 'P14714', 'Q6XFQ2', 'Q10CQ8', 'Q6XFQ3', 'Q10MG9', 'B4YB07', 'P14712', 'P42497', 'Q6XFQ1', 'B4YB10', 'Q10DU0', 'P19862', 'C1PHB8', 'P42498', 'Q6XFQ4', 'A0A1D6GGX9', 'A0A0R0HSY5', 'P42499', 'Q8IAU3', 'P42500', 'P07451', 'P16015', 'Q3THS6', 'F1QYU7', 'Q7ZW04', 'P9WIK7', 'P13444', 'P43166', 'P14141', 'I1LB04', 'P23686', 'Q92051', 'Q1RLT0', 'Q9SJL8', 'I1LZ62', 'I1MRW7', 'P10659', 'Q9LUT2', 'P18298', 'P40320', 'O43570', 'P00918', 'Q9ERQ8', 'P00920', 'Q58EF9', 'I1MRW8', 'I1L8X2', 'Q91X83', 'I1LVJ0', 'I1MHR0', 'P9WFY7', 'I1JPQ2', 'Q0DKY4', 'Q16790', 'B5DFG6', 'B2RZ61', 'Q9D6N1', 'O60198', 'Q9LGU6', 'Q8N1Q1', 'B4FIE9', 'P93438', 'Q9ULX7', 'I1JQV8', 'I1NBG9', 'P27139', 'P0A8I8', 'A0A1D8PF68', 'P31153', 'P19358', 'Q2FZ43', 'P0A873', 'A0A1D6FHW2', 'Q54F07', 'B8A068', 'O50394', 'P50306', 'Q46803', 'F7FL00', 'Q2G252', 'Q59034', 'Q8VHB5', 'Q27522', 'P50305', 'P17562', 'Q9V9Y8', 'A0A1D6HJ15', 'I1L041', 'Q66HG6', 'A2IBE2', 'Q9Y2D0', 'Q4CSC4', 'Q9LXB4', 'Q9WVT6', 'E7FBE2', 'Q4KLI2', 'P22748', 'Q00266', 'E9AHK3', 'Q57977', 'P0AGJ7', 'I1MFS5', 'O14214', 'Q64444', 'Q27504', 'Q9V9Y6', 'P23589', 'Q9FGI9', 'R4GDY8', 'Q9QZA0', 'A8KB74', 'A0A1D6FUX8', 'Q59Q39', 'Q8CI85', 'Q8TBZ6', 'P35218', 'Q92979', 'P23280', 'Q5T280', 'Q9NP92', 'O95831', 'Q7L0Y3', 'Q8NBA8', 'Q8NDH3', 'Q7RTV0', 'Q14CN2', 'O95905', 'Q9BZE1', 'A8K7I4', 'Q9UJK0', 'Q6PF06', 'Q13395', 'Q9UQC9', 'Q7RTX0', 'Q6IN84', 'Q8TE23', 'Q6UXX5', 'Q96RN1', 'Q9NYU1', 'Q9NYU2', 'Q9NQV8', 'Q13075', 'Q9UPR5', 'P57103', 'Q12791', 'P32418', 'A8MYU2', 'Q8N5C7', 'Q9NVC6', 'Q9HC36']

# List PDB IDs failed to download (there is not PDB files for these in the Protein Data Bank)
# failed_ids = ['5oql', '5wyk', '5nrl', '6rxz', '6rxy', '6rxx', '6rxv', '6rxt', '6qx9', '5z57', '5z56', '6gaw']

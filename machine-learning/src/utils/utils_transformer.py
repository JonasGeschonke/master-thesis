import os
import requests
import torch
import librosa
import soundfile as sf
import numpy as np
import mlflow

from mlflow.client import MlflowClient
from pathlib import Path
from torch import Tensor
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import butter, sosfilt
from tqdm import tqdm

from dataset.textgrid import TextGrid

PHONEME_INVENTORY = ['', 'a', 'aj', 'aw', 'aː', 'b', 'bʲ', 'c', 'cʰ', 'cʷ', 'd', 'dʒ', 'dʲ', 'd̪', 'e', 'ej', 'eː', 'f', 'fʲ', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'kʷ', 'l', 'm', 'mʲ', 'm̩', 'n', 'o', 'ow', 'oː', 'p', 'pʰ', 'pʲ', 's', 'spn', 't', 'tʃ', 'tʰ', 'tʲ', 'tʷ', 't̪', 'u', 'uː', 'v', 'vʲ', 'w', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔ', 'ɔj', 'ɖ', 'ə', 'əw', 'ɚ', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɝ', 'ɟ', 'ɡ', 'ɪ', 'ɫ', 'ɲ', 'ɹ', 'ɾ', 'ʃ', 'ʈ', 'ʈʲ', 'ʉ', 'ʉː', 'ʊ', 'ʋ', 'ʎ', 'ʒ', 'θ']
PHONEME_TO_INDEX = {phoneme: idx for idx, phoneme in enumerate(PHONEME_INVENTORY)}
MEMBRANE_UUIDS = [
    '00ac9d67-2e51-4fbe-be8a-d7a4635af78b',
    '010eeb4f-279f-432b-855c-eb3d6d8f64f4',
    '01cc0049-940b-4039-88d1-11d3fcceaa9c',
    '026dc899-a9c8-4436-b471-2e9fcda3d4d4',
    '04ee3e3f-a652-4fd8-881d-b039c7b34f7e',
    '051ff658-c8a9-4553-bfa4-de42344fc689',
    '053f2118-a353-4d73-9b5d-8f9d6f997e54',
    '06a08640-951b-4829-96b8-712c7b6e5679',
    '06a88475-6da0-483b-8ec4-71ce94affce1',
    '06e0f93f-d54a-4f7b-b0ca-b9a89ece30fa',
    '070cd78c-0071-4fba-adf3-743cad5b6d27',
    '0784c935-8823-4745-af2c-4b39fe061c71',
    '07f2af6f-4245-4fed-9177-b8f0ecad6240',
    '08349e61-c503-44e9-ac65-bbe65b18e3fa',
    '099c7da1-bb56-4450-9efb-798028eeb62f',
    '09f88f82-b4c0-411d-b0b6-d107107273f5',
    '0a6f9803-223d-422d-b2d6-bc3efa57291e',
    '0b3745b8-70fb-4d72-b2ee-ff81f15f2159',
    '0ce79e89-64db-40e4-96d2-81d381077ceb',
    '0d09ac3b-94b6-4311-aa83-540ecde4f6a3',
    '0d0dcb90-dded-4c97-a779-0bb3e28c6ae0',
    '0e5b28ad-496b-4d4f-9111-416f68d26883',
    '0f85c7b5-52d6-4aad-bb97-1b227b3aa563',
    '104468cf-3744-4416-92d0-5e6f501f8037',
    '111d7799-b14e-498e-be1d-86df1eff2e80',
    '143619cc-52ee-4252-af04-d6fd7b69ccef',
    '1468c562-a083-4244-a97e-8a2346d3fb70',
    '14a95f42-98d2-48ed-8e4b-753135426886',
    '15200b4f-d909-484b-820e-d81e974f229d',
    '1564c5c0-821b-481f-be25-68a2365f88f5',
    '15f4377f-56e6-4ec5-bd48-137f14438c01',
    '165ea51b-7898-441c-920b-9596d865d034',
    '169c25a8-d0d4-4da2-9ef4-4bc0e25650ad',
    '17cf5646-c4d5-473d-9c42-b8309d7ff498',
    '189d78a9-be29-4181-b36e-6e9b5c76fc14',
    '1948246d-74ce-47bd-bedd-acc5cb7b2fa0',
    '19600005-2865-4fe8-9072-126c6a2e1067',
    '1962977e-b6cf-458d-b95d-0be89ac5b5b2',
    '1cb81b32-10b5-47a6-bede-c09289cf1f22',
    '1ce713ab-1188-4c29-9c5e-4f2821cfb4b5',
    '232661f8-379a-4a4d-978d-7d3f3b828a49',
    '23f9797a-852f-4aa0-b9d6-503cf2a9dbd1',
    '241e779b-6e33-4fbf-ac6b-524869448cb8',
    '256657b1-4c0d-46b1-bf62-8ca44f152113',
    '265de728-5634-4629-af97-52542dd5dccb',
    '26fada18-81c8-4602-9bf5-cee60c9e95a6',
    '273d9fef-8994-4b56-9f50-61e2345dff47',
    '27883497-bab7-41ba-beda-7118d8376d0d',
    '2816e9d6-8539-40d8-8ccd-434e35e42d65',
    '28df42ac-96cf-46b3-8ed9-01050e969e0b',
    '29782332-d496-4d86-a5cf-8f967f9b9677',
    '29c8cfb8-7693-48b2-af51-cfed915d0415',
    '2a87880f-a41d-462c-b6ca-4d84c0de5bf9',
    '2a945448-58e8-42c6-a371-5076751b7acb',
    '2a9e2b6b-cf75-4906-a7b2-3a75b909f11a',
    '2ae166bd-f975-491a-af08-b528d1350675',
    '2c301e8b-37ae-44ce-b32b-664f45e8c483',
    '2c3d1b76-953e-43a5-8dfa-78537975850a',
    '2e158966-74d2-48bf-a592-e3ad3271447a',
    '2f6fafdb-61c0-4022-844a-1dd1ca9a1610',
    '2f83f91e-e482-4689-9285-c71d5b0682cb',
    '2fddce42-f5d3-4952-9fde-2e0785f71204',
    '30d0a211-5860-47b6-a85c-47c360231b3f',
    '31a6f56a-45e3-4dea-9a92-d7190aba0114',
    '32c3cc3e-67f5-47e4-8e75-7870280fc4c2',
    '34934042-b29f-4ca9-b373-9612ee521775',
    '35d92cca-b435-4d6b-bfa9-6ddb80a111a2',
    '3709e515-3179-4316-8163-e7c1245aaaba',
    '38c3aa1c-c563-4000-a046-a3bddc5bbdf3',
    '3a757216-2553-42df-a03d-2e61e9ce1d4a',
    '3be6c386-227d-4d29-bf87-fd9347257d01',
    '3d2ce256-761d-40c7-a0fe-d4f480f9ad51',
    '3f5bad6c-6762-4aab-a1a7-b6095357ade2',
    '4134d727-1215-4b65-b830-f50cc3ca5e81',
    '42af7529-9b17-4f0a-81be-fe7ca18c0bff',
    '44f80cd4-8592-43ca-9c34-135dbc9e3cb4',
    '454f2b4a-ee31-4410-9ff2-12d81182e82f',
    '47eaad93-ba4a-4760-8b20-cfe6ae07659c',
    '47fd9d3e-aa92-4244-a37d-d9c0550ca48a',
    '4855f1df-7b3d-44a4-9ec2-7e8436bbb45d',
    '492a289f-33f2-4cc7-a6e4-675cad300ddb',
    '49c66866-431f-4b9f-bd51-370c671f4f2d',
    '49c6a592-4284-484d-8970-ea1c7090a734',
    '4a457f2f-b97b-461f-acc4-4e097d843e77',
    '4bf40073-fa6a-411c-99d1-7595242e16d4',
    '4cc73510-0c52-4b5a-af60-f370ab2313da',
    '4df34c99-d89d-40bd-8afc-a82461af0e39',
    '4e1d0945-658f-4dc6-b33d-e447a2959f6f',
    '4f7bde9d-ee4d-4f5e-bd6a-3923e50a5e90',
    '4faab01e-e01a-4559-b163-896aa216148c',
    '5108fa78-1d78-4017-80ee-34bdab17be37',
    '534d01d6-8141-44ef-a8a9-d80f3a6ff387',
    '542983e0-a019-477a-a79c-188ead52fdd2',
    '54510ed6-2329-4d26-89d3-d4880754d404',
    '58e4b49b-1101-4cef-ae87-4ebab837587e',
    '58f42678-070f-422b-9456-2a5606122b14',
    '598f79e7-a5ff-4415-88d2-34510e18bce9',
    '59963b98-0aba-4617-835c-168ca58d6b2b',
    '5ac6279c-c5c2-462b-a9d2-644fa1eb19c0',
    '5aed843e-144d-40fb-84ad-014e16518dda',
    '5cd78c6d-7e61-4489-9b24-2d0e710a7e14',
    '5d3cb6bd-4b59-4afc-9a3c-e4bd8ac32d73',
    '5d85c877-c5d6-440b-b1c3-275d7f0ba7bb',
    '5d85ca1f-9f5a-4a60-b092-1317f01a83d2',
    '5e67fcb0-e047-41f5-96f7-4988d42a3e48',
    '601027ec-d5fd-4adf-abbb-d1a15e787a5e',
    '60510849-87bc-4216-aed3-b1720c8e24bd',
    '62fa0864-8a6b-43cd-86b5-c542b8a393f1',
    '63b4df53-384d-46d3-9544-ae83e165e474',
    '642528ae-07db-4c48-aa56-9752e72c5762',
    '64a4f5eb-2a2f-41a9-b7cb-b9176c870e3c',
    '64f38c97-347d-41c8-a04b-064c05657e8e',
    '657b635b-b95d-4cb9-9eeb-2dfa6c2a0ef9',
    '65ee3210-f752-4d40-a49c-c745612f34be',
    '66158751-44be-4231-bccf-30a7385baac4',
    '66c00cc6-84e6-4641-b353-39c6d36d701c',
    '67abc685-5d91-42cc-8f83-f73a245d6da4',
    '68e02c0b-ef84-4826-a8e0-035a85f329b7',
    '6a423206-5032-4035-8e69-2c32bc5c6140',
    '6bba8de9-91a5-48d3-bf0c-fcbd420664f7',
    '6c45368d-d680-4932-af31-ddaf2ea7d041',
    '6c818b2a-0071-4031-b80c-c60702117d2a',
    '6dedb7b0-2c1e-447f-b917-b5f51930817e',
    '6e8411ed-8cda-4912-94cd-62782dad2017',
    '6f0eaf74-fd17-4854-86da-0804c488d90e',
    '70e20167-0220-4617-8b29-e220fc5076c1',
    '71ded9ae-6842-48f0-8056-265e3a1bf0e6',
    '721465bb-32bb-4088-918a-87c5406d0004',
    '723a35bc-efa8-4d85-9a67-811cc7236245',
    '727e0464-f218-4d6a-8680-e5a637f9bf8e',
    '74c6c444-5bab-4922-8495-2751272ca307',
    '76c9fcf9-df47-44f3-b6f8-a7e1625aa87e',
    '7a459f30-9c0c-4449-b458-440e5e2b17ef',
    '7a76b9a6-739a-4092-a3e7-585220b94b90',
    '7d29a86e-bc08-4c84-a155-b32a5cfe4dec',
    '7dec232d-bf7f-47f3-b02c-96a84e680682',
    '7e0f5389-ad5f-4b29-be12-332b0cca3054',
    '7ec16c08-9933-4197-a35f-a41cc594cba4',
    '7ee8d329-b49a-43e4-bf54-e0794a47b209',
    '7f0e6920-1a3d-4d90-b5ae-efdd5afd6b11',
    '7fecca1c-f411-4ff5-ad77-7c95845c6a04',
    '817cd86e-0349-4439-9c02-8cd8bc168d61',
    '8456d656-f71f-4fc9-9219-1a4be5797ab2',
    '84fa9cad-5af3-466a-bd23-55fdba87b1df',
    '8510a844-f306-454f-8506-541fe15f5e8c',
    '863302cf-1983-4fc9-a036-57b4845dccf9',
    '87260b23-ed5f-4d90-b487-2e48335e92a3',
    '879097ae-6fe2-4768-add6-69f73c3d1b40',
    '8887cbd4-fe7d-4865-bce1-cabc64b8f1c2',
    '889366dd-2a59-43cd-91d2-2fa1ff606c6f',
    '88ae7185-93aa-49a3-98fe-70ee066a9e0a',
    '89b5e34d-612a-4cc8-a0c8-4c9045b354c2',
    '8aa2c518-f462-4007-a898-3b0cce6d279c',
    '8ae7406e-a63a-41d5-90b8-08210d9a0ee5',
    '8b33c297-20b5-4680-92f7-69abd4f8a707',
    '8c5a1299-f77a-42d2-9e44-02835b3417be',
    '8c63c671-3cde-48d0-806c-1f5c38b17878',
    '8ca89e0d-668b-4448-9e4c-0e36aff3c0d1',
    '8ddc2d3a-54ba-445f-bdca-b0f423c731c4',
    '8fc9cecc-3835-45ca-a896-3bcfd7086279',
    '8ffefd76-1e71-4b87-9fc8-6d6119732dab',
    '90164296-533f-4bdd-b46e-b65b0680256f',
    '90285376-56b8-4509-bf18-ee1e9cc0e992',
    '90698b8e-154a-415a-9aa0-2af3d19da914',
    '9088eec4-6f54-4820-8726-5dc8c85a0843',
    '9164b2c5-f45b-4bf0-9d5e-ea72b9c4051e',
    '92ec8d57-d1e5-41c7-a4c4-5285dee858fc',
    '93550a95-8df6-4bc8-b323-05333a6c8256',
    '9462ef59-0161-4e5d-a2e3-9d603f396d59',
    '9531bf0f-9521-4802-92fc-b83a8cf9d792',
    '95b9f9e9-6426-4259-bdc7-8637f974792d',
    '96bfd96a-d89a-4a59-8df6-cd5a7c1b9397',
    '974e25c4-f93c-4533-a613-8089aa108eea',
    '9a464fac-e3ca-40ad-a6e2-cc22a04919f7',
    '9a7120c8-0c77-4c0f-b101-799a3921efbd',
    '9c0db24e-7254-492d-8070-bb848c5d7ae2',
    '9c829caa-9ef7-488e-851c-d7ad2ecb11bd',
    '9c8eaa87-c1da-41f9-bd03-f6cbc860fa02',
    '9d38b44e-8392-4f68-a467-ba6989d79d7c',
    'a06ed05f-14ff-4a74-83ad-2a0810c5f54d',
    'a087a1a9-3715-4312-bddc-538f47b0aac7',
    'a178b39c-c17b-48f3-bba9-8c248d655256',
    'a1c0535e-8b87-43ba-9f09-7f5da999b583',
    'a2ce61ca-01d7-4dbe-b3ec-7d8da55b21fd',
    'a4512453-ca7e-4899-939f-5e10f4950862',
    'a453edb6-6275-461d-b4c7-1d4680460517',
    'a4f3eb42-94d4-4fe6-a12d-a8b7a8201757',
    'a51c95ae-6ab1-4829-a140-254adef3a74b',
    'a537c350-d722-4fac-9b1c-70b7a7d0033e',
    'a53d9e09-4f59-451e-91e2-09d631f6611b',
    'a7abdade-f2ca-444d-87f7-701da1e0414e',
    'a9108967-d732-4088-84f9-0231b72ec85f',
    'a96fe509-d243-4e3d-8fe4-808542a36831',
    'aaa34c32-6609-47bb-8a30-09ce24880257',
    'aaa6e1ab-3dbf-4e26-88de-0ef10353eb46',
    'ab757dbc-a83f-441e-a1d3-258abb614ef7',
    'abe2216c-3b77-47c7-853d-1dda0a897f6c',
    'ad26b592-e53e-4fdc-8271-3adfb364cdc5',
    'ad4ce956-7849-4625-9853-6069b01aa7d4',
    'ad4ff2e4-247a-42ca-9655-c2e907bcd489',
    'ad823e45-85fd-4b4b-8b8c-2466d4b1b564',
    'ad9e9a46-7a17-4cf5-a57b-b248243132ad',
    'ae6e9b73-0aae-4d01-bad1-be535d0b69de',
    'af3c479d-984e-4529-9035-ecefcbfa59a4',
    'afe262cc-b219-4ef1-9566-57d5698c90a4',
    'b032b2d1-ddee-4c4d-ab67-20b6eb9c21f2',
    'b0c042c4-93e9-4720-a8f6-86ac1cb845b2',
    'b0e75073-0a88-48cd-a228-b08a012af0e1',
    'b257c395-8f40-4002-b0f9-4083e4ee05ea',
    'b45a8e68-9a3b-4087-a8b6-504b9af85090',
    'b4f5aebb-62e0-4922-b9f1-27045e8dac7c',
    'b6855388-f86d-4222-87f6-f861ce775bba',
    'b6d7afce-d608-4023-9532-78040e40b9e4',
    'b71fa725-6adb-45bd-91cb-38d067e05186',
    'b7ba7290-3224-4250-834f-1ad84f30ec50',
    'b9ce436e-5d98-4445-8be8-ad1bf2ad6bb9',
    'ba3dc115-98aa-4744-a99c-00cb5debd932',
    'bbddfe0a-af08-4817-a1a4-a9047c675b41',
    'bc8fad4c-086d-46b5-833b-3159efff2e8b',
    'bcb4885e-6b2e-4a6f-b807-116f1a14020e',
    'bd3cc0db-5b28-4df5-8261-49f341267c54',
    'bd90e037-b151-4b90-aaed-e6cf086a78fd',
    'be5fecfe-ba97-49bf-8143-d4a72eeb7ecc',
    'be77433a-1902-4004-9a65-131694d35317',
    'beaab56b-4d79-40d6-8230-f51927041f48',
    'bf2edee4-1d64-4295-9ae3-c0649e3c4761',
    'bf54d6b0-2345-44b0-96bb-284f9c336cf5',
    'bf84840f-7824-46e3-8ae9-7dae4d78082d',
    'c279dc8b-c927-4196-8a57-91d5a2b59e72',
    'c5404b8f-1121-4bfe-a6a4-c7e19f4fe554',
    'cb42a2ae-d51e-495c-be38-74b12d4b7a7a',
    'cdd1ef6d-2ea1-4132-b019-56d4e3e1bbdb',
    'cfd8657b-e5fb-40f7-b303-d89b70ddfe09',
    'd03f7f1d-9ad4-4eb8-91f9-f85c9b8deaf4',
    'd03f805b-8d05-4014-9c22-23fec1a5f592',
    'd0b9c8d5-1772-4bdd-bbf4-3de0a1fadb6a',
    'd2aece76-ace4-4956-b78c-48972348b46f',
    'd351bb2a-12fe-4a60-9549-41adb8bff2ea',
    'd4e3504d-c74f-478b-8292-81f8ce01762f',
    'd5050210-099a-4426-b437-5753cefeaf3a',
    'd77add93-b24d-49a5-840e-b38d9f0057b0',
    'd7cb3bee-694b-449d-93aa-8d5b70eefe0c',
    'd90ed194-6732-411e-939b-3a1d01ec8255',
    'd95f1feb-30e8-4b87-9624-affddc8fa193',
    'dac10ffe-1be7-498c-aba0-26c66b8a7cf4',
    'dbb5ca9c-9c23-4a00-b514-4c3880e7d277',
    'dc068450-5b29-4f41-a57f-dbcc038a1ff8',
    'dce0975f-2f8d-47a5-aee4-353827c70984',
    'ddeeee44-bb0a-4f2a-8a5f-19596d2008ee',
    'de2998e5-785f-4589-8666-df099b532d12',
    'de6d4e6c-583e-4e40-8d2c-a858809b87d7',
    'defaee7f-8992-4baf-abd5-337174b4cca2',
    'dfbc9438-86f2-4674-9a02-568d8ad36589',
    'e0177f61-e813-459d-9b1d-40608a81c626',
    'e029d5d2-9156-45ce-9502-55dc26139db9',
    'e047aadf-c48c-477d-9cdd-cb851c0135d6',
    'e3332c6a-569d-4887-8db4-036ebdb6a020',
    'e39c2f8f-b4ac-4fd8-8611-f104f143497a',
    'e436e8a6-508b-405d-9c15-969dde600733',
    'e652652f-a11e-4d5f-a2b0-d7d9d94e1586',
    'e661eb93-862a-4e6f-bb9e-f0a741a0da81',
    'e6db4a80-d98a-4fed-9bf4-fa829c828417',
    'e85831a5-5969-466d-93f7-aebf0ad86e07',
    'eaaa841c-329b-43e1-8eeb-19dc1d626242',
    'eb0dd511-1a16-4e96-b646-63fca74cabc5',
    'eb71daf1-1e71-43e6-82f8-87388ac5ebc5',
    'ec462733-f71e-4a47-9856-9950420e3517',
    'ec6e06b6-2891-403a-9f91-b8a51a88c354',
    'ece4c221-a219-42ef-aa27-814ac572967a',
    'ece566c0-f301-4303-a362-0750ff2de4d3',
    'ed5b02ef-b414-4ac7-b132-9e5066c57e86',
    'ef740ef7-9df4-4d11-b65d-27d73947bc23',
    'f11e10ea-8334-4d17-ae97-5a5fae8c985d',
    'f1fb26c0-f7fe-4fa0-ab7d-0da972b7c59d',
    'f2821061-ced8-49f3-8727-7c2efc9ef620',
    'f287fd89-5e2f-48ce-a533-60f55b29f8d1',
    'f3377b6b-eaa5-4bcc-acac-19122b7a0a76',
    'f6429860-eb77-448e-bda7-8f7e7526b58d',
    'f68350a2-0ef9-4022-b8c0-11fc746c7845',
    'f6af9cdb-6a3d-47f6-8b91-dcedacd15754',
    'f783b738-b7ee-4680-ac24-439b62a72d10',
    'f87975c3-f99a-429a-8b13-fd51b055066e',
    'f8ff69b8-0aba-40eb-b94d-a5ffde56529a',
    'f96f7624-c663-4cdc-8e55-f00c0fd8f2d9',
    'f9fd06b4-0326-4da5-98a5-a5b1c04c340e',
    'fce5ac75-b36c-4025-b1e1-8410afff1f96',
    'fd7a8a2f-91b4-446a-81df-74bea93c2bcf',
    'fd9ac7ac-9ff0-4260-9d0f-56bebe5c4ab1',
    'fda78ead-c6f9-45a7-99ba-5202364ebd46',
    'fde478f3-8d38-43a7-ba36-fa905c5cfcbf',
    'fe65e9f3-dd38-4af9-83a4-7b023794786c',
    'ffab77cb-8fe5-40eb-a31f-32c7de828580',
]

def download_data(backend_url: str, data_path: Path, voiced_path: Path, unvoiced_path: Path, debug: bool = False, textgrid_path: Path = Path('textgrids')) -> None:
    def download_data(url: str, filepath: Path) -> None:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def download_textgrid(url: str, filepath: Path) -> None:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "w") as f:
            f.write(response.text)

    recordings = requests.get(backend_url + "/recordings").json()
    if debug:
        print(f"Found {len(recordings)} recordings on the backend")
    uuids = [recording['uuid'] for recording in recordings]
    data_path.mkdir(exist_ok=True)
    unvoiced_path.mkdir(exist_ok=True)
    voiced_path.mkdir(exist_ok=True)
    textgrid_path.mkdir(exist_ok=True)
    audio_files = [
        {"voiced_url": f'{backend_url}/audio/voiced/{uuid}', "unvoiced_url": f'{backend_url}/audio/unvoiced/{uuid}', "filename": f'{uuid}.flac'} for uuid in uuids
    ]
    for audio_file in tqdm(audio_files, desc="Downloading audio files"):
        filepath_voiced = voiced_path / audio_file["filename"]
        filepath_unvoiced = unvoiced_path / audio_file["filename"]
        if not filepath_voiced.exists():
            download_data(audio_file["voiced_url"], filepath_voiced)
        if not filepath_unvoiced.exists():
            download_data(audio_file["unvoiced_url"], filepath_unvoiced)

    for uuid in tqdm(uuids, desc="Downloading TextGrid files"):
        textgrid_url = f'{backend_url}/textgrids/{uuid}'
        textgrid_file_path = textgrid_path / f'{uuid}.TextGrid'
        if not textgrid_file_path.exists():
            download_textgrid(textgrid_url, textgrid_file_path)

def download_data_for_uuids(backend_url: str, uuids: list[str], voiced_path: Path, unvoiced_path: Path, textgrid_path: Path, debug: bool = False) -> None:
    """Download only the recordings matching the given UUIDs."""
    def download_data(url: str, filepath: Path) -> None:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def download_textgrid(url: str, filepath: Path) -> None:
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "w") as f:
            f.write(response.text)

    voiced_path.mkdir(exist_ok=True)
    unvoiced_path.mkdir(exist_ok=True)
    textgrid_path.mkdir(exist_ok=True)

    for uuid in tqdm(uuids, desc="Downloading membrane audio files"):
        voiced_file = voiced_path / f'{uuid}.flac'
        unvoiced_file = unvoiced_path / f'{uuid}.flac'
        if not voiced_file.exists():
            download_data(f'{backend_url}/audio/voiced/{uuid}', voiced_file)
        if not unvoiced_file.exists():
            download_data(f'{backend_url}/audio/unvoiced/{uuid}', unvoiced_file)

    for uuid in tqdm(uuids, desc="Downloading membrane TextGrid files"):
        textgrid_file_path = textgrid_path / f'{uuid}.TextGrid'
        if not textgrid_file_path.exists():
            download_textgrid(f'{backend_url}/textgrids/{uuid}', textgrid_file_path)

    if debug:
        print(f"Downloaded {len(uuids)} membrane recordings into {voiced_path.parent}")

def normalize_device_string(device: str) -> str:
    if not isinstance(device, str) or not device.startswith('cuda'):
        return device
    if device == 'cuda':
        return device

    parts = device.split(':', 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return device

    physical_index = int(parts[1])
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible:
        return device

    visible_indices = []
    for item in visible.split(','):
        item = item.strip()
        if item == '':
            continue
        try:
            visible_indices.append(int(item))
        except ValueError:
            return device

    if physical_index in visible_indices:
        local_index = visible_indices.index(physical_index)
        return f'cuda:{local_index}'
    return device

def setup_manual_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_torch(device: str) -> str:
    normalized_device = normalize_device_string(device)
    if isinstance(normalized_device, str) and normalized_device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.set_device(torch.device(normalized_device))
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    return normalized_device

def setup_mlflow(server_url: str, experiment_name: str):
    mlflow.set_tracking_uri(server_url)
    return mlflow.set_experiment(experiment_name)

def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, scheduler=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    # Include scheduler state if provided
    if scheduler is not None:
        try:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        except Exception:
            pass

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file_name = f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path / checkpoint_file_name)

def normalize_volume(audio):
    rms = librosa.feature.rms(y=audio)
    max_rms = rms.max() + 0.01
    target_rms = 0.2
    audio = audio * (target_rms/max_rms)
    max_val = np.abs(audio).max()
    if max_val > 1.0: # this shouldn't happen too often with the target_rms of 0.2
        audio = audio / max_val
    return audio

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis_cache = {}
hann_window_cache = {}
 
def BIGVGAN_mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
) -> torch.Tensor:
    """Calculate the mel spectrogram of an input signal."""
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"
    global mel_basis_cache, hann_window_cache

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(
        y.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis_cache, hann_window_cache
    if fmax not in mel_basis_cache:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis_cache[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window_cache[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis_cache[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def _resample_to_target_sr(audio: np.ndarray, orig_sr: int, target_sr: int = 44100) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def _lowpass_remove_carrier(audio: np.ndarray, cutoff: float = 16000, sr: float = 44100, order: int = 6) -> np.ndarray:
    """Apply a Butterworth low-pass filter to remove the 19 kHz ultrasound carrier."""
    sos = butter(order, cutoff, btype='low', fs=sr, output='sos')
    return sosfilt(sos, audio)

def load_audio(filename, start=None, end=None, max_frames=None, renormalize_volume=False, target_sr: int = 44100):
    audio, r = sf.read(filename)

    if len(audio.shape) > 1:
        audio = audio[:, 0]  # select first channel of stereo audio
    if start is not None or end is not None:
        audio = audio[start:end]

    if renormalize_volume:
        audio = normalize_volume(audio)

    audio = _resample_to_target_sr(audio, r, target_sr)
    audio = np.clip(audio, -1, 1)

    # Remove the 19 kHz ultrasound carrier so the GT mel only contains speech-relevant frequencies.
    audio = _lowpass_remove_carrier(audio, cutoff=16000, sr=target_sr)

    pytorch_mspec = BIGVGAN_mel_spectrogram(
        torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
        1024,
        128,
        target_sr,
        256,
        1024,
        0,
        None,
        center=False,
    )
    mspec = pytorch_mspec.squeeze(0).T
    if max_frames is not None and mspec.shape[0] > max_frames:
        mspec = mspec[:max_frames, :]
    return mspec


def load_lp_mel_from_audio(filename, target_sr: int = 44100) -> Tensor:
    """Load audio and compute a low-pass (0-17 kHz) mel spectrogram with the correct processing order: STFT magnitude → mel filterbank → log compression."""
    audio, r = sf.read(filename)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = normalize_volume(audio)
    audio = _resample_to_target_sr(audio, r, target_sr)
    audio = np.clip(audio, -1, 1)

    pytorch_mspec = BIGVGAN_mel_spectrogram(
        torch.tensor(audio, dtype=torch.float32).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=target_sr,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=17000,
        center=True,
    )
    return pytorch_mspec.squeeze(0).T

def load_stft_audio(filename, target_sr: int = 44100) -> Tensor:
    audio, r = sf.read(filename)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = _resample_to_target_sr(audio, r, target_sr)
    audio = np.clip(audio, -1, 1)
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    window = torch.hann_window(1024).to('cpu')
    stft = torch.stft(
        audio_tensor,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=window,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = torch.abs(stft)
    spec_tensor = spectral_normalize_torch(magnitude)
    spec_tensor = spec_tensor.T
    return spec_tensor

def load_stft_complex_audio(filename, target_sr: int = 44100, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024) -> Tensor:
    audio, r = sf.read(filename)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = _resample_to_target_sr(audio, r, target_sr)
    audio = np.clip(audio, -1, 1)
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    window = torch.hann_window(win_length).to('cpu')
    stft = torch.stft(
        audio_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return stft.T

def extract_ultrasound_reflection_features(
    filename,
    carrier_freq: float = 19000.0,
    bandwidth: float = 512.0,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> Tensor:
    """Extract narrowband reflection features from an ultrasound carrier recording."""
    stft = load_stft_complex_audio(filename, target_sr=sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    num_time_steps, num_freq_bins = stft.shape
    freq_resolution = sample_rate / n_fft
    center_bin = int(round(carrier_freq / freq_resolution))
    half_band = int(round(bandwidth / freq_resolution))
    low_bin = max(0, center_bin - half_band)
    high_bin = min(num_freq_bins - 1, center_bin + half_band)
    if low_bin >= high_bin:
        low_bin = max(0, center_bin - 1)
        high_bin = min(num_freq_bins - 1, center_bin + 1)

    narrowband = stft[:, low_bin:high_bin + 1]
    static_carrier = narrowband.mean(dim=0, keepdim=True)
    residual = narrowband - static_carrier

    residual_env = residual.abs()
    residual_energy = residual_env.sum(dim=-1)
    carrier_energy = narrowband.abs().sum(dim=-1)
    energy_ratio = residual_energy / (carrier_energy + 1e-8)

    phase = torch.angle(residual)
    phase_diff = torch.zeros_like(phase)
    phase_diff[1:] = phase[1:] - phase[:-1]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    mean_instant_freq = phase_diff.mean(dim=-1)

    flux = torch.zeros(num_time_steps, dtype=residual_env.dtype, device=residual_env.device)
    flux[1:] = (residual_env[1:] - residual_env[:-1]).abs().mean(dim=-1)

    centroid_freqs = torch.linspace(low_bin * freq_resolution, high_bin * freq_resolution, steps=narrowband.shape[1], device=residual_env.device)
    centroid = (residual_env * centroid_freqs).sum(dim=-1) / (residual_env.sum(dim=-1) + 1e-8)

    std_env = residual_env.std(dim=-1)
    mean_env = residual_env.mean(dim=-1)

    features = torch.stack(
        [residual_energy, energy_ratio, mean_env, std_env, centroid, mean_instant_freq, flux],
        dim=-1,
    )

    # Normalize each feature dimension to zero mean / unit variance per sample.
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True)
    features = (features - feature_mean) / (feature_std + 1e-8)
    return features

def textgrid_to_phoneme_sequence(textgrid_path: Path, spectrogram: Tensor, sample_rate: int = 44100, hop_length: int = 256) -> Tensor:
    num_frames = spectrogram.shape[0]
    try:
        textgrid = TextGrid.load(textgrid_path)
    except Exception as e:
        print(f"TextGrid file {textgrid_path} could not be loaded: {e}")
        return torch.zeros(num_frames, dtype=torch.long)

    phoneme_item = textgrid.tiers[1]

    phoneme_labels = torch.zeros(num_frames, dtype=torch.long)

    for start, end, label in phoneme_item.simple_transcript:
        normalized_label = label.strip()
        start_frame = int(np.floor(float(start) * sample_rate / hop_length))
        end_frame = int(np.ceil(float(end) * sample_rate / hop_length))
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(start_frame, min(end_frame, num_frames))
        phoneme_index = PHONEME_TO_INDEX.get(normalized_label, 0)
        phoneme_labels[start_frame:end_frame] = phoneme_index

    return phoneme_labels

def split_spectrogram(spec: Tensor, freq_cutoff: int, sample_rate: int) -> tuple[Tensor, Tensor]:
    num_freq_bins = spec.shape[1]
    freq_ceiling = sample_rate / 2
    cut_index = int(freq_cutoff / (freq_ceiling / num_freq_bins))
    low_pass: Tensor = spec[:, :cut_index - 1]
    high_pass: Tensor = spec[:, cut_index:]

    return low_pass, high_pass

def convert_to_mel_spectrogram(linear_spec: torch.Tensor, num_mel_bins: int = 128) -> torch.Tensor:
    """Converts a linear spectrogram to a Mel spectrogram."""
    num_freq_bins, num_time_steps = linear_spec.shape
    sample_rate = 44100
    num_mel_bins = min(num_mel_bins, num_freq_bins)
    n_fft = (num_freq_bins - 1) * 2
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mel_bins, fmin=0.0, fmax=17000.0)
    fb = torch.from_numpy(mel_basis).to(device=linear_spec.device, dtype=linear_spec.dtype)
    mel_spec = torch.matmul(fb, linear_spec)

    return mel_spec
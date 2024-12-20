# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: hudes.proto
# Protobuf Python Version: 5.28.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    28,
    1,
    '',
    'hudes.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0bhudes.proto\x12\x05hudes\"\'\n\nDimAndStep\x12\x0b\n\x03\x64im\x18\x01 \x02(\x05\x12\x0c\n\x04step\x18\x02 \x02(\x02\"\x95\x01\n\x06\x43onfig\x12\x0c\n\x04seed\x18\x01 \x02(\x05\x12\x16\n\x0e\x64ims_at_a_time\x18\x02 \x02(\x05\x12\x16\n\x0emesh_grid_size\x18\x03 \x02(\x05\x12\x16\n\x0emesh_step_size\x18\x04 \x02(\x02\x12\x12\n\nmesh_grids\x18\x05 \x02(\x05\x12\x12\n\nbatch_size\x18\x06 \x02(\x05\x12\r\n\x05\x64type\x18\x07 \x02(\t\"\x85\x01\n\x11TrainLossAndPreds\x12\x12\n\ntrain_loss\x18\x01 \x02(\x01\x12\r\n\x05preds\x18\x02 \x03(\x02\x12\x13\n\x0bpreds_shape\x18\x03 \x03(\x05\x12\x18\n\x10\x63onfusion_matrix\x18\x04 \x03(\x02\x12\x1e\n\x16\x63onfusion_matrix_shape\x18\x05 \x03(\x05\"\x1b\n\x07ValLoss\x12\x10\n\x08val_loss\x18\x01 \x02(\x01\"\xca\x01\n\rBatchExamples\x12\'\n\x04type\x18\x01 \x02(\x0e\x32\x19.hudes.BatchExamples.Type\x12\t\n\x01n\x18\x02 \x02(\x05\x12\x12\n\ntrain_data\x18\x03 \x03(\x02\x12\x18\n\x10train_data_shape\x18\x04 \x03(\x05\x12\x14\n\x0ctrain_labels\x18\x05 \x03(\x02\x12\x1a\n\x12train_labels_shape\x18\x06 \x03(\x05\x12\x11\n\tbatch_idx\x18\x08 \x02(\x05\"\x12\n\x04Type\x12\n\n\x06IMG_BW\x10\x00\"\x8d\x05\n\x07\x43ontrol\x12!\n\x04type\x18\x01 \x02(\x0e\x32\x13.hudes.Control.Type\x12*\n\x0e\x64ims_and_steps\x18\xc8\x01 \x03(\x0b\x32\x11.hudes.DimAndStep\x12\x1e\n\x06\x63onfig\x18\xc9\x01 \x01(\x0b\x32\r.hudes.Config\x12\x37\n\x14train_loss_and_preds\x18\xca\x01 \x01(\x0b\x32\x18.hudes.TrainLossAndPreds\x12!\n\x08val_loss\x18\xcb\x01 \x01(\x0b\x32\x0e.hudes.ValLoss\x12\x14\n\x0brequest_idx\x18\xcc\x01 \x01(\x05\x12-\n\x0e\x62\x61tch_examples\x18\xcd\x01 \x01(\x0b\x32\x14.hudes.BatchExamples\x12\x1a\n\x11mesh_grid_results\x18\xcf\x01 \x03(\x02\x12\x18\n\x0fmesh_grid_shape\x18\xd0\x01 \x03(\x05\x12\x12\n\tsgd_steps\x18\xd1\x01 \x01(\x05\x12\x18\n\x0ftotal_sgd_steps\x18\xd2\x01 \x01(\x05\"\x8d\x02\n\x04Type\x12\x10\n\x0c\x43ONTROL_DIMS\x10\x00\x12 \n\x1c\x43ONTROL_TRAIN_LOSS_AND_PREDS\x10\x01\x12\x14\n\x10\x43ONTROL_VAL_LOSS\x10\x02\x12\x1a\n\x16\x43ONTROL_BATCH_EXAMPLES\x10\x03\x12\x15\n\x11\x43ONTROL_FULL_LOSS\x10\x04\x12\x16\n\x12\x43ONTROL_NEXT_BATCH\x10\x65\x12\x15\n\x11\x43ONTROL_NEXT_DIMS\x10\x66\x12\x12\n\x0e\x43ONTROL_CONFIG\x10g\x12\x1c\n\x18\x43ONTROL_MESHGRID_RESULTS\x10h\x12\x14\n\x10\x43ONTROL_SGD_STEP\x10i\x12\x11\n\x0c\x43ONTROL_QUIT\x10\x85\x07')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'hudes_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_DIMANDSTEP']._serialized_start=22
  _globals['_DIMANDSTEP']._serialized_end=61
  _globals['_CONFIG']._serialized_start=64
  _globals['_CONFIG']._serialized_end=213
  _globals['_TRAINLOSSANDPREDS']._serialized_start=216
  _globals['_TRAINLOSSANDPREDS']._serialized_end=349
  _globals['_VALLOSS']._serialized_start=351
  _globals['_VALLOSS']._serialized_end=378
  _globals['_BATCHEXAMPLES']._serialized_start=381
  _globals['_BATCHEXAMPLES']._serialized_end=583
  _globals['_BATCHEXAMPLES_TYPE']._serialized_start=565
  _globals['_BATCHEXAMPLES_TYPE']._serialized_end=583
  _globals['_CONTROL']._serialized_start=586
  _globals['_CONTROL']._serialized_end=1239
  _globals['_CONTROL_TYPE']._serialized_start=970
  _globals['_CONTROL_TYPE']._serialized_end=1239
# @@protoc_insertion_point(module_scope)

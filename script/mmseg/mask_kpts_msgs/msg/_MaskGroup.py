# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from mask_kpts_msgs/MaskGroup.msg. Do not edit."""
import codecs
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct

import mask_kpts_msgs.msg
import sensor_msgs.msg
import std_msgs.msg

class MaskGroup(genpy.Message):
  _md5sum = "8907c665f5ae0d37e1d20499fd77db01"
  _type = "mask_kpts_msgs/MaskGroup"
  _has_header = True  # flag to mark the presence of a Header object
  _full_text = """std_msgs/Header header
MaskKpts[] objects
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

================================================================================
MSG: mask_kpts_msgs/MaskKpts
string label
uint16 track_id
sensor_msgs/Image mask
Keypoint bbox_tl
Keypoint bbox_br
Keypoint[] kpts_curr
Keypoint[] kpts_last

================================================================================
MSG: sensor_msgs/Image
# This message contains an uncompressed image
# (0, 0) is at top-left corner of image
#

Header header        # Header timestamp should be acquisition time of image
                     # Header frame_id should be optical frame of camera
                     # origin of frame should be optical center of camera
                     # +x should point to the right in the image
                     # +y should point down in the image
                     # +z should point into to plane of the image
                     # If the frame_id here and the frame_id of the CameraInfo
                     # message associated with the image conflict
                     # the behavior is undefined

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

# The legal values for encoding are in file src/image_encodings.cpp
# If you want to standardize a new string format, join
# ros-users@lists.sourceforge.net and send an email proposing a new encoding.

string encoding       # Encoding of pixels -- channel meaning, ordering, size
                      # taken from the list of strings in include/sensor_msgs/image_encodings.h

uint8 is_bigendian    # is this data bigendian?
uint32 step           # Full row length in bytes
uint8[] data          # actual matrix data, size is (step * rows)

================================================================================
MSG: mask_kpts_msgs/Keypoint
float32 x
float32 y
float32 z
"""
  __slots__ = ['header','objects']
  _slot_types = ['std_msgs/Header','mask_kpts_msgs/MaskKpts[]']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       header,objects

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(MaskGroup, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.objects is None:
        self.objects = []
    else:
      self.header = std_msgs.msg.Header()
      self.objects = []

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      length = len(self.objects)
      buff.write(_struct_I.pack(length))
      for val1 in self.objects:
        _x = val1.label
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = val1.track_id
        buff.write(_get_struct_H().pack(_x))
        _v1 = val1.mask
        _v2 = _v1.header
        _x = _v2.seq
        buff.write(_get_struct_I().pack(_x))
        _v3 = _v2.stamp
        _x = _v3
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v2.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = _v1
        buff.write(_get_struct_2I().pack(_x.height, _x.width))
        _x = _v1.encoding
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = _v1
        buff.write(_get_struct_BI().pack(_x.is_bigendian, _x.step))
        _x = _v1.data
        length = len(_x)
        # - if encoded as a list instead, serialize as bytes instead of string
        if type(_x) in [list, tuple]:
          buff.write(struct.Struct('<I%sB'%length).pack(length, *_x))
        else:
          buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _v4 = val1.bbox_tl
        _x = _v4
        buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
        _v5 = val1.bbox_br
        _x = _v5
        buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
        length = len(val1.kpts_curr)
        buff.write(_struct_I.pack(length))
        for val2 in val1.kpts_curr:
          _x = val2
          buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
        length = len(val1.kpts_last)
        buff.write(_struct_I.pack(length))
        for val2 in val1.kpts_last:
          _x = val2
          buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.objects is None:
        self.objects = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.objects = []
      for i in range(0, length):
        val1 = mask_kpts_msgs.msg.MaskKpts()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.label = str[start:end].decode('utf-8', 'rosmsg')
        else:
          val1.label = str[start:end]
        start = end
        end += 2
        (val1.track_id,) = _get_struct_H().unpack(str[start:end])
        _v6 = val1.mask
        _v7 = _v6.header
        start = end
        end += 4
        (_v7.seq,) = _get_struct_I().unpack(str[start:end])
        _v8 = _v7.stamp
        _x = _v8
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v7.frame_id = str[start:end].decode('utf-8', 'rosmsg')
        else:
          _v7.frame_id = str[start:end]
        _x = _v6
        start = end
        end += 8
        (_x.height, _x.width,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v6.encoding = str[start:end].decode('utf-8', 'rosmsg')
        else:
          _v6.encoding = str[start:end]
        _x = _v6
        start = end
        end += 5
        (_x.is_bigendian, _x.step,) = _get_struct_BI().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        _v6.data = str[start:end]
        _v9 = val1.bbox_tl
        _x = _v9
        start = end
        end += 12
        (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
        _v10 = val1.bbox_br
        _x = _v10
        start = end
        end += 12
        (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.kpts_curr = []
        for i in range(0, length):
          val2 = mask_kpts_msgs.msg.Keypoint()
          _x = val2
          start = end
          end += 12
          (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
          val1.kpts_curr.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.kpts_last = []
        for i in range(0, length):
          val2 = mask_kpts_msgs.msg.Keypoint()
          _x = val2
          start = end
          end += 12
          (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
          val1.kpts_last.append(val2)
        self.objects.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self
      buff.write(_get_struct_3I().pack(_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs))
      _x = self.header.frame_id
      length = len(_x)
      if python3 or type(_x) == unicode:
        _x = _x.encode('utf-8')
        length = len(_x)
      buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
      length = len(self.objects)
      buff.write(_struct_I.pack(length))
      for val1 in self.objects:
        _x = val1.label
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = val1.track_id
        buff.write(_get_struct_H().pack(_x))
        _v11 = val1.mask
        _v12 = _v11.header
        _x = _v12.seq
        buff.write(_get_struct_I().pack(_x))
        _v13 = _v12.stamp
        _x = _v13
        buff.write(_get_struct_2I().pack(_x.secs, _x.nsecs))
        _x = _v12.frame_id
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = _v11
        buff.write(_get_struct_2I().pack(_x.height, _x.width))
        _x = _v11.encoding
        length = len(_x)
        if python3 or type(_x) == unicode:
          _x = _x.encode('utf-8')
          length = len(_x)
        buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _x = _v11
        buff.write(_get_struct_BI().pack(_x.is_bigendian, _x.step))
        _x = _v11.data
        length = len(_x)
        # - if encoded as a list instead, serialize as bytes instead of string
        if type(_x) in [list, tuple]:
          buff.write(struct.Struct('<I%sB'%length).pack(length, *_x))
        else:
          buff.write(struct.Struct('<I%ss'%length).pack(length, _x))
        _v14 = val1.bbox_tl
        _x = _v14
        buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
        _v15 = val1.bbox_br
        _x = _v15
        buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
        length = len(val1.kpts_curr)
        buff.write(_struct_I.pack(length))
        for val2 in val1.kpts_curr:
          _x = val2
          buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
        length = len(val1.kpts_last)
        buff.write(_struct_I.pack(length))
        for val2 in val1.kpts_last:
          _x = val2
          buff.write(_get_struct_3f().pack(_x.x, _x.y, _x.z))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    codecs.lookup_error("rosmsg").msg_type = self._type
    try:
      if self.header is None:
        self.header = std_msgs.msg.Header()
      if self.objects is None:
        self.objects = None
      end = 0
      _x = self
      start = end
      end += 12
      (_x.header.seq, _x.header.stamp.secs, _x.header.stamp.nsecs,) = _get_struct_3I().unpack(str[start:end])
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      start = end
      end += length
      if python3:
        self.header.frame_id = str[start:end].decode('utf-8', 'rosmsg')
      else:
        self.header.frame_id = str[start:end]
      start = end
      end += 4
      (length,) = _struct_I.unpack(str[start:end])
      self.objects = []
      for i in range(0, length):
        val1 = mask_kpts_msgs.msg.MaskKpts()
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          val1.label = str[start:end].decode('utf-8', 'rosmsg')
        else:
          val1.label = str[start:end]
        start = end
        end += 2
        (val1.track_id,) = _get_struct_H().unpack(str[start:end])
        _v16 = val1.mask
        _v17 = _v16.header
        start = end
        end += 4
        (_v17.seq,) = _get_struct_I().unpack(str[start:end])
        _v18 = _v17.stamp
        _x = _v18
        start = end
        end += 8
        (_x.secs, _x.nsecs,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v17.frame_id = str[start:end].decode('utf-8', 'rosmsg')
        else:
          _v17.frame_id = str[start:end]
        _x = _v16
        start = end
        end += 8
        (_x.height, _x.width,) = _get_struct_2I().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        if python3:
          _v16.encoding = str[start:end].decode('utf-8', 'rosmsg')
        else:
          _v16.encoding = str[start:end]
        _x = _v16
        start = end
        end += 5
        (_x.is_bigendian, _x.step,) = _get_struct_BI().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        start = end
        end += length
        _v16.data = str[start:end]
        _v19 = val1.bbox_tl
        _x = _v19
        start = end
        end += 12
        (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
        _v20 = val1.bbox_br
        _x = _v20
        start = end
        end += 12
        (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.kpts_curr = []
        for i in range(0, length):
          val2 = mask_kpts_msgs.msg.Keypoint()
          _x = val2
          start = end
          end += 12
          (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
          val1.kpts_curr.append(val2)
        start = end
        end += 4
        (length,) = _struct_I.unpack(str[start:end])
        val1.kpts_last = []
        for i in range(0, length):
          val2 = mask_kpts_msgs.msg.Keypoint()
          _x = val2
          start = end
          end += 12
          (_x.x, _x.y, _x.z,) = _get_struct_3f().unpack(str[start:end])
          val1.kpts_last.append(val2)
        self.objects.append(val1)
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_2I = None
def _get_struct_2I():
    global _struct_2I
    if _struct_2I is None:
        _struct_2I = struct.Struct("<2I")
    return _struct_2I
_struct_3I = None
def _get_struct_3I():
    global _struct_3I
    if _struct_3I is None:
        _struct_3I = struct.Struct("<3I")
    return _struct_3I
_struct_3f = None
def _get_struct_3f():
    global _struct_3f
    if _struct_3f is None:
        _struct_3f = struct.Struct("<3f")
    return _struct_3f
_struct_BI = None
def _get_struct_BI():
    global _struct_BI
    if _struct_BI is None:
        _struct_BI = struct.Struct("<BI")
    return _struct_BI
_struct_H = None
def _get_struct_H():
    global _struct_H
    if _struct_H is None:
        _struct_H = struct.Struct("<H")
    return _struct_H

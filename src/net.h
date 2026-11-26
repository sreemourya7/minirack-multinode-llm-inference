// net.h - the wire protocol: how workers (and the client) talk to each other.
//
// TCP is a byte stream with no message boundaries, so every message is
// "framed": a fixed 32-byte header that says what the message is and how many
// payload bytes follow. The receiver reads the header, then exactly that many
// bytes.
//
//   offset  size  field
//   0       4     magic       0x314B524D ("MRK1") - catches garbage / wrong peer
//   4       4     type        MsgType below
//   8       4     flags       FLAG_* bits
//   12      4     reserved    0
//   16      8     request_id  echoed back in the reply
//   24      8     payload_len bytes of payload that follow
//
// All integers and floats are little-endian (x86 and ARM both are).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mat.h"

namespace mr {

enum MsgType : uint32_t {
  MSG_TOKENS = 1,  // client -> first worker.   payload: u32 n, then n int32 token ids
  MSG_HIDDEN = 2,  // worker -> next worker.    payload: u32 rows, u32 cols, float32 data
  MSG_LOGITS = 3,  // reply, last worker -> ... -> client. same layout as MSG_HIDDEN
  MSG_ERROR = 4,   // reply. payload: UTF-8 text
};

enum : uint32_t {
  FLAG_LAST_ONLY = 1u << 0,  // only return logits for the final token
};

struct Msg {
  uint32_t type = 0;
  uint32_t flags = 0;
  uint64_t request_id = 0;
  std::vector<uint8_t> payload;
};

// --- sockets ---------------------------------------------------------------
int listen_on(int port);                             // returns a listening fd, throws on failure
int accept_one(int listen_fd);                       // blocks; returns a connected fd or -1
int connect_to(const std::string& host, int port);   // returns fd or -1
void close_fd(int fd);

// --- framed messages -------------------------------------------------------
bool send_msg(int fd, const Msg& m);  // false if the peer went away
bool recv_msg(int fd, Msg& m);        // false on clean close, error, or a bad header

// --- payload encoding ------------------------------------------------------
std::vector<uint8_t> encode_mat(const Mat& m);
Mat decode_mat(const std::vector<uint8_t>& payload);
std::vector<int32_t> decode_tokens(const std::vector<uint8_t>& payload);
Msg make_error(uint64_t request_id, const std::string& text);

}  // namespace mr

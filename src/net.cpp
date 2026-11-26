#include "net.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace mr {

namespace {

constexpr uint32_t kMagic = 0x314B524D;            // "MRK1"
constexpr uint64_t kMaxPayload = 1ull << 30;       // 1 GiB: refuse absurd sizes

struct Header {
  uint32_t magic;
  uint32_t type;
  uint32_t flags;
  uint32_t seq;
  uint64_t request_id;
  uint64_t payload_len;
};
static_assert(sizeof(Header) == 32, "header must be exactly 32 bytes");

// send() and recv() may move fewer bytes than asked. These loop until done.
bool send_all(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  while (n > 0) {
    // MSG_NOSIGNAL: if the peer died, return an error instead of killing us
    // with SIGPIPE.
    const ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
    if (k < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

bool recv_all(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  while (n > 0) {
    const ssize_t k = ::recv(fd, p, n, 0);
    if (k == 0) return false;  // peer closed the connection
    if (k < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += k;
    n -= static_cast<size_t>(k);
  }
  return true;
}

void set_nodelay(int fd) {
  // Turn off Nagle's algorithm. By default TCP may hold back small writes
  // hoping to batch them; for request/reply traffic that only adds latency.
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
}

}  // namespace

int listen_on(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("socket() failed");
  int one = 1;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);  // allow quick restarts on the same port
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0) {
    ::close(fd);
    throw std::runtime_error("bind() failed on port " + std::to_string(port) + ": " + std::strerror(errno));
  }
  if (::listen(fd, 16) != 0) {
    ::close(fd);
    throw std::runtime_error("listen() failed");
  }
  return fd;
}

int accept_one(int listen_fd) {
  for (;;) {
    const int fd = ::accept(listen_fd, nullptr, nullptr);
    if (fd >= 0) {
      set_nodelay(fd);
      return fd;
    }
    if (errno != EINTR) return -1;
  }
}

int connect_to(const std::string& host, int port) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0) return -1;
  int fd = -1;
  for (addrinfo* p = res; p; p = p->ai_next) {
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd < 0) continue;
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
    ::close(fd);
    fd = -1;
  }
  ::freeaddrinfo(res);
  if (fd >= 0) set_nodelay(fd);
  return fd;
}

void close_fd(int fd) {
  if (fd >= 0) ::close(fd);
}

bool send_msg(int fd, const Msg& m) {
  Header h{};
  h.magic = kMagic;
  h.type = m.type;
  h.flags = m.flags;
  h.seq = m.seq;
  h.request_id = m.request_id;
  h.payload_len = m.payload.size();
  if (!send_all(fd, &h, sizeof h)) return false;
  if (!m.payload.empty() && !send_all(fd, m.payload.data(), m.payload.size())) return false;
  return true;
}

bool recv_msg(int fd, Msg& m) {
  Header h;
  if (!recv_all(fd, &h, sizeof h)) return false;
  if (h.magic != kMagic || h.payload_len > kMaxPayload) return false;
  m.type = h.type;
  m.flags = h.flags;
  m.seq = h.seq;
  m.request_id = h.request_id;
  m.payload.resize(h.payload_len);
  if (h.payload_len > 0 && !recv_all(fd, m.payload.data(), m.payload.size())) return false;
  return true;
}

std::vector<uint8_t> encode_mat(const Mat& m) {
  const uint32_t dims[2] = {static_cast<uint32_t>(m.rows), static_cast<uint32_t>(m.cols)};
  std::vector<uint8_t> out(sizeof dims + m.size() * sizeof(float));
  std::memcpy(out.data(), dims, sizeof dims);
  std::memcpy(out.data() + sizeof dims, m.data.data(), m.size() * sizeof(float));
  return out;
}

Mat decode_mat(const std::vector<uint8_t>& payload) {
  uint32_t dims[2];
  if (payload.size() < sizeof dims) throw std::runtime_error("matrix payload too short");
  std::memcpy(dims, payload.data(), sizeof dims);
  const uint64_t want = static_cast<uint64_t>(dims[0]) * dims[1] * sizeof(float);
  if (payload.size() - sizeof dims != want) throw std::runtime_error("matrix payload size does not match its shape");
  Mat m(static_cast<int>(dims[0]), static_cast<int>(dims[1]));
  std::memcpy(m.data.data(), payload.data() + sizeof dims, want);
  return m;
}

std::vector<int32_t> decode_tokens(const std::vector<uint8_t>& payload) {
  uint32_t n;
  if (payload.size() < sizeof n) throw std::runtime_error("token payload too short");
  std::memcpy(&n, payload.data(), sizeof n);
  if (payload.size() - sizeof n != static_cast<uint64_t>(n) * sizeof(int32_t)) {
    throw std::runtime_error("token payload size does not match its count");
  }
  std::vector<int32_t> ids(n);
  std::memcpy(ids.data(), payload.data() + sizeof n, n * sizeof(int32_t));
  return ids;
}

Msg make_error(uint64_t request_id, const std::string& text) {
  Msg m;
  m.type = MSG_ERROR;
  m.request_id = request_id;
  m.payload.assign(text.begin(), text.end());
  return m;
}

}  // namespace mr
